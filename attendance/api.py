"""REST API for attendance records, enrollment and reporting.

    uvicorn attendance.api:app --reload
    open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import csv
import io
import os

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from attendance import service
from attendance.config import Settings, get_settings
from attendance.db import init_db, make_engine, make_session_factory
from attendance.enroll import enroll_image, sample_counts, train_from_disk
from attendance.face import FaceDetector, FaceRecognizer

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class StudentIn(BaseModel):
    roll_no: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    name: str = Field(min_length=1, max_length=120)
    class_name: str = ""


class SessionIn(BaseModel):
    class_name: str = Field(min_length=1, max_length=64)
    subject: str = ""


class ManualMark(BaseModel):
    roll_no: str


def student_out(s, faces=0):
    return {"roll_no": s.roll_no, "name": s.name, "class_name": s.class_name, "face_samples": faces}


def session_out(s):
    return {"id": s.id, "class_name": s.class_name, "subject": s.subject,
            "started_at": s.started_at.isoformat(), "ended_at": s.ended_at.isoformat() if s.ended_at else None}


async def read_image(upload: UploadFile):
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image larger than 5 MB")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, f"{upload.filename} is not a readable image")
    return image


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    engine = make_engine(settings.db_url)
    init_db(engine)
    Session = make_session_factory(engine)
    detector = FaceDetector(face_size=settings.face_size)
    state = {"recognizer": None}

    def load_recognizer():
        if state["recognizer"] is None:
            if not os.path.exists(settings.model_path):
                raise HTTPException(409, "No trained model yet: enroll students and POST /model/train")
            state["recognizer"] = FaceRecognizer(settings.match_threshold).load(settings.model_path)
        return state["recognizer"]

    def get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI(title="Face Recognition Attendance API", version="1.0.0")

    @app.exception_handler(service.NotFound)
    async def not_found(_, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(service.Conflict)
    async def conflict(_, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        return {"status": "ok", "model_trained": os.path.exists(settings.model_path)}

    # ---- students -----------------------------------------------------------
    @app.post("/students", status_code=201)
    def create_student(body: StudentIn, db=Depends(get_db)):
        return student_out(service.add_student(db, body.roll_no, body.name, body.class_name))

    @app.get("/students")
    def list_students(class_name: str | None = None, db=Depends(get_db)):
        counts = sample_counts(settings.faces_dir)
        return [student_out(s, counts.get(s.roll_no, 0)) for s in service.list_students(db, class_name)]

    @app.get("/students/{roll_no}")
    def get_student(roll_no: str, db=Depends(get_db)):
        return student_out(service.get_student(db, roll_no), sample_counts(settings.faces_dir).get(roll_no, 0))

    @app.delete("/students/{roll_no}", status_code=204)
    def delete_student(roll_no: str, db=Depends(get_db)):
        service.delete_student(db, roll_no)

    @app.post("/students/{roll_no}/faces", status_code=201)
    async def upload_faces(roll_no: str, files: list[UploadFile] = File(...), db=Depends(get_db)):
        """Enroll one or more photos; each must contain exactly one face."""
        service.get_student(db, roll_no)
        saved, rejected = 0, []
        for f in files:
            try:
                enroll_image(await read_image(f), roll_no, settings.faces_dir, detector)
                saved += 1
            except (ValueError, HTTPException) as e:
                rejected.append({"file": f.filename, "reason": getattr(e, "detail", str(e))})
        return {"roll_no": roll_no, "saved": saved, "rejected": rejected,
                "total_samples": sample_counts(settings.faces_dir).get(roll_no, 0)}

    # ---- model --------------------------------------------------------------
    @app.post("/model/train")
    def train():
        try:
            recognizer, n_students, n_faces = train_from_disk(
                settings.faces_dir, settings.model_path, settings.match_threshold)
        except ValueError as e:
            raise HTTPException(409, str(e))
        state["recognizer"] = recognizer
        return {"students": n_students, "faces": n_faces}

    # ---- sessions & attendance ----------------------------------------------
    @app.post("/sessions", status_code=201)
    def create_session(body: SessionIn, db=Depends(get_db)):
        return session_out(service.start_session(db, body.class_name, body.subject))

    @app.get("/sessions/{session_id}")
    def get_session(session_id: int, db=Depends(get_db)):
        return session_out(service.get_session(db, session_id))

    @app.post("/sessions/{session_id}/end")
    def end_session(session_id: int, db=Depends(get_db)):
        return session_out(service.end_session(db, session_id))

    @app.post("/sessions/{session_id}/recognize")
    async def recognize(session_id: int, file: UploadFile = File(...), db=Depends(get_db)):
        """Recognise every face in one photo (e.g. from a classroom kiosk) and mark known students."""
        service.get_session(db, session_id)
        recognizer = load_recognizer()
        faces = []
        for det in detector.detect(await read_image(file)):
            roll_no, distance = recognizer.predict(det.face)
            entry = {"box": det.box, "roll_no": roll_no, "distance": round(distance, 1), "marked": False}
            if roll_no is not None:
                _, created = service.mark_present(db, session_id, roll_no, confidence=distance)
                entry["marked"] = created
            faces.append(entry)
        return {"faces_found": len(faces), "faces": faces}

    @app.post("/sessions/{session_id}/attendance", status_code=201)
    def mark_manually(session_id: int, body: ManualMark, db=Depends(get_db)):
        """Manual override, e.g. for a student the camera missed."""
        record, created = service.mark_present(db, session_id, body.roll_no, method="manual")
        return {"roll_no": body.roll_no, "created": created, "marked_at": record.marked_at.isoformat()}

    @app.get("/sessions/{session_id}/report")
    def report(session_id: int, format: str = Query("json", pattern="^(json|csv)$"), db=Depends(get_db)):
        data = service.session_report(db, session_id)
        if format == "csv":
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=["roll_no", "name", "status", "marked_at", "method"])
            writer.writeheader()
            writer.writerows(data["students"])
            return PlainTextResponse(buf.getvalue(), media_type="text/csv", headers={
                "Content-Disposition": f'attachment; filename="session_{session_id}.csv"'})
        return data

    @app.get("/classes/{class_name}/summary")
    def class_summary(class_name: str, db=Depends(get_db)):
        return service.student_summary(db, class_name)

    return app


app = create_app() if os.environ.get("ATTENDANCE_NO_AUTOAPP") != "1" else None
