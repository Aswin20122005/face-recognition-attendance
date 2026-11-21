"""Attendance business logic, independent of the camera and the web API."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from attendance.models import AttendanceRecord, ClassSession, Student


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


# ---- students -----------------------------------------------------------------

def add_student(db: Session, roll_no: str, name: str, class_name: str = "") -> Student:
    if db.scalar(select(Student).where(Student.roll_no == roll_no)):
        raise Conflict(f"Student {roll_no} already exists")
    student = Student(roll_no=roll_no, name=name, class_name=class_name)
    db.add(student)
    db.commit()
    return student


def get_student(db: Session, roll_no: str) -> Student:
    student = db.scalar(select(Student).where(Student.roll_no == roll_no))
    if student is None:
        raise NotFound(f"Student {roll_no} not found")
    return student


def list_students(db: Session, class_name: str | None = None) -> list[Student]:
    query = select(Student).order_by(Student.roll_no)
    if class_name:
        query = query.where(Student.class_name == class_name)
    return list(db.scalars(query))


def delete_student(db: Session, roll_no: str) -> None:
    db.delete(get_student(db, roll_no))
    db.commit()


# ---- sessions -----------------------------------------------------------------

def start_session(db: Session, class_name: str, subject: str = "") -> ClassSession:
    session = ClassSession(class_name=class_name, subject=subject)
    db.add(session)
    db.commit()
    return session


def get_session(db: Session, session_id: int) -> ClassSession:
    session = db.get(ClassSession, session_id)
    if session is None:
        raise NotFound(f"Session {session_id} not found")
    return session


def end_session(db: Session, session_id: int) -> ClassSession:
    session = get_session(db, session_id)
    if session.ended_at is None:
        session.ended_at = datetime.now(timezone.utc)
        db.commit()
    return session


# ---- marking ------------------------------------------------------------------

def mark_present(db: Session, session_id: int, roll_no: str, method: str = "face",
                 confidence: float | None = None) -> tuple[AttendanceRecord, bool]:
    """Mark a student present. Returns (record, created).

    Marking an already-present student is not an error: the existing record is
    returned with created=False, so the camera loop can call this repeatedly.
    """
    session = get_session(db, session_id)
    if session.ended_at is not None:
        raise Conflict(f"Session {session_id} has ended")
    student = get_student(db, roll_no)

    existing = db.scalar(select(AttendanceRecord).where(
        AttendanceRecord.session_id == session_id, AttendanceRecord.student_id == student.id))
    if existing:
        return existing, False

    record = AttendanceRecord(session_id=session_id, student_id=student.id, method=method, confidence=confidence)
    db.add(record)
    try:
        db.commit()
    except IntegrityError:  # marked concurrently by another request
        db.rollback()
        existing = db.scalar(select(AttendanceRecord).where(
            AttendanceRecord.session_id == session_id, AttendanceRecord.student_id == student.id))
        return existing, False
    return record, True


# ---- reports ------------------------------------------------------------------

def session_report(db: Session, session_id: int) -> dict:
    """Present and absent students for one session (absent = in the class, not marked)."""
    session = get_session(db, session_id)
    marked = {r.student_id: r for r in session.records}
    roster = list_students(db, session.class_name)
    rows = []
    for s in roster:
        r = marked.get(s.id)
        rows.append({
            "roll_no": s.roll_no, "name": s.name, "status": "present" if r else "absent",
            "marked_at": r.marked_at.isoformat() if r else None, "method": r.method if r else None,
        })
    present = sum(1 for row in rows if row["status"] == "present")
    return {
        "session_id": session.id, "class_name": session.class_name, "subject": session.subject,
        "started_at": session.started_at.isoformat(), "present": present, "absent": len(rows) - present,
        "total": len(rows), "students": rows,
    }


def student_summary(db: Session, class_name: str) -> list[dict]:
    """Attendance percentage per student over all sessions of a class."""
    n_sessions = db.scalar(select(func.count()).select_from(ClassSession)
                           .where(ClassSession.class_name == class_name)) or 0
    counts = dict(db.execute(
        select(AttendanceRecord.student_id, func.count())
        .join(ClassSession, AttendanceRecord.session_id == ClassSession.id)
        .where(ClassSession.class_name == class_name)
        .group_by(AttendanceRecord.student_id)
    ).all())
    out = []
    for s in list_students(db, class_name):
        attended = counts.get(s.id, 0)
        out.append({
            "roll_no": s.roll_no, "name": s.name, "attended": attended, "sessions": n_sessions,
            "percentage": round(100 * attended / n_sessions, 1) if n_sessions else 0.0,
        })
    return out
