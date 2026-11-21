import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from attendance.api import create_app
from attendance.config import Settings
from tests.synthetic import draw_face


def png(image):
    return cv2.imencode(".png", image)[1].tobytes()


@pytest.fixture
def client(tmp_path):
    settings = Settings(data_dir=str(tmp_path), database_url="", match_threshold=80)
    return TestClient(create_app(settings))


def add(client, roll, name="Student", class_name="CSE-A"):
    return client.post("/students", json={"roll_no": roll, "name": name, "class_name": class_name})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_trained": False}


def test_student_crud_and_validation(client):
    assert add(client, "CS001", "Asha").status_code == 201
    assert add(client, "CS001").status_code == 409
    assert add(client, "../x").status_code == 422
    assert client.get("/students/CS001").json()["name"] == "Asha"
    assert client.get("/students/NOPE").status_code == 404
    assert client.delete("/students/CS001").status_code == 204
    assert client.get("/students").json() == []


def test_upload_faces_reports_rejected_photos(client):
    add(client, "CS001")
    files = [("files", ("a.png", png(draw_face(0)), "image/png")),
             ("files", ("b.png", png(draw_face(1)), "image/png")),
             ("files", ("blank.png", png(np.zeros((200, 200, 3), np.uint8)), "image/png")),
             ("files", ("bad.png", b"not an image", "image/png"))]
    body = client.post("/students/CS001/faces", files=files).json()
    assert body["saved"] == 2 and body["total_samples"] == 2
    assert {r["file"] for r in body["rejected"]} == {"blank.png", "bad.png"}
    assert client.get("/students/CS001").json()["face_samples"] == 2


def test_train_without_faces_is_a_conflict(client):
    assert client.post("/model/train").status_code == 409


def test_recognize_before_training_is_a_conflict(client):
    session = client.post("/sessions", json={"class_name": "CSE-A"}).json()
    r = client.post(f"/sessions/{session['id']}/recognize", files={"file": ("x.png", png(draw_face(0)), "image/png")})
    assert r.status_code == 409


def test_full_flow_enroll_train_recognize_report(client):
    add(client, "CS001", "Asha")
    add(client, "CS002", "Ravi")
    files = [("files", (f"{k}.png", png(draw_face(k)), "image/png")) for k in range(5)]
    client.post("/students/CS001/faces", files=files)
    assert client.post("/model/train").json() == {"students": 1, "faces": 5}

    session = client.post("/sessions", json={"class_name": "CSE-A", "subject": "DBMS"}).json()
    photo = {"file": ("class.png", png(draw_face(3)), "image/png")}
    result = client.post(f"/sessions/{session['id']}/recognize", files=photo).json()
    assert result["faces_found"] == 1
    assert result["faces"][0]["roll_no"] == "CS001" and result["faces"][0]["marked"] is True
    # a second photo doesn't mark twice
    again = client.post(f"/sessions/{session['id']}/recognize", files=photo).json()
    assert again["faces"][0]["marked"] is False

    manual = client.post(f"/sessions/{session['id']}/attendance", json={"roll_no": "CS002"})
    assert manual.status_code == 201

    report = client.get(f"/sessions/{session['id']}/report").json()
    assert (report["present"], report["absent"]) == (2, 0)
    methods = {s["roll_no"]: s["method"] for s in report["students"]}
    assert methods == {"CS001": "face", "CS002": "manual"}

    csv_text = client.get(f"/sessions/{session['id']}/report?format=csv").text
    assert csv_text.splitlines()[0] == "roll_no,name,status,marked_at,method"

    client.post(f"/sessions/{session['id']}/end")
    late = client.post(f"/sessions/{session['id']}/attendance", json={"roll_no": "CS001"})
    assert late.status_code == 409

    summary = {r["roll_no"]: r["percentage"] for r in client.get("/classes/CSE-A/summary").json()}
    assert summary == {"CS001": 100.0, "CS002": 100.0}


def test_unknown_session_is_404(client):
    assert client.get("/sessions/999/report").status_code == 404


def test_dashboard_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "session-form" in r.text
    assert client.get("/static/app.js").status_code == 200
