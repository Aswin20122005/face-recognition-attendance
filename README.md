# Student Attendance via Face Recognition 📸

[![CI](https://github.com/Aswin20122005/face-recognition-attendance/actions/workflows/ci.yml/badge.svg)](https://github.com/Aswin20122005/face-recognition-attendance/actions/workflows/ci.yml)

An automated attendance system that replaces manual roll-call: **OpenCV** detects and recognises students' faces in real time from a webcam (or in an uploaded classroom photo), attendance is stored in a **SQL database**, and a **REST API** plus a small web dashboard provide records and reports.

## Features

- **Real-time recognition** from a webcam with on-screen labels (unknown / recognised / marked present)
- **Photo mode**: upload or snap a classroom photo and every known face is marked
- **Reliable marking**: a student must be recognised in several consecutive frames before being marked, and can only be marked once per session (enforced by a database constraint)
- **Enrollment** from the webcam or a folder of photos; photos with zero or several faces are rejected
- **SQL storage**: students, class sessions and attendance records with SQLAlchemy (SQLite by default; MySQL or PostgreSQL via `DATABASE_URL`)
- **Reports**: present/absent per session (JSON or CSV) and attendance percentage per student
- **Manual override** for students the camera missed
- **REST API** with interactive docs at `/docs`, and a web dashboard at `/`

## How it works

```
webcam frame / photo
      │
      ▼
Haar cascade face detection ──► crop · grayscale · 200×200 · histogram equalisation
      │
      ▼
LBPH recogniser ──► roll number + distance (above MATCH_THRESHOLD = unknown)
      │
      ▼
confirmation: N consecutive frames (CONFIRM_FRAMES)
      │
      ▼
SQL: attendance_records (unique per session + student)
      │
      ▼
REST API / dashboard / CLI reports
```

Both models ship with OpenCV, so nothing needs to be downloaded. LBPH trains in seconds on a CPU, needs only 10 to 20 photos per student, and can be retrained when a student is added.

## Quick start

```bash
git clone https://github.com/Aswin20122005/face-recognition-attendance.git
cd face-recognition-attendance
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Command line (webcam)

```bash
python -m attendance.cli add-student CS001 "Asha Kumar" --class CSE-A
python -m attendance.cli enroll CS001 --samples 20         # captures 20 photos from the webcam
python -m attendance.cli enroll CS002 --images photos/ravi # or use existing photos
python -m attendance.cli train
python -m attendance.cli take --class CSE-A --subject DBMS # live attendance; press q to finish
python -m attendance.cli report --class CSE-A --csv cse-a.csv
```

### API and dashboard

```bash
uvicorn attendance.api:create_app --factory --reload
# dashboard:  http://127.0.0.1:8000/
# API docs:   http://127.0.0.1:8000/docs
```

Or with Docker:

```bash
docker compose up --build        # data persists in the attendance-data volume
```

## REST API

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Status and whether a model is trained |
| POST | `/students` | Add a student `{roll_no, name, class_name}` |
| GET | `/students?class_name=` | List students with their face-sample counts |
| GET / DELETE | `/students/{roll_no}` | Get or delete a student (and their records) |
| POST | `/students/{roll_no}/faces` | Upload enrollment photos (multipart, one face each) |
| POST | `/model/train` | Train the recogniser on all enrolled faces |
| POST | `/sessions` | Start a class session `{class_name, subject}` |
| POST | `/sessions/{id}/recognize` | Upload a photo; marks every recognised student |
| POST | `/sessions/{id}/attendance` | Manually mark `{roll_no}` present |
| POST | `/sessions/{id}/end` | End the session (no further marking) |
| GET | `/sessions/{id}/report?format=json\|csv` | Present / absent list |
| GET | `/classes/{class_name}/summary` | Attendance percentage per student |

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `ATTENDANCE_DATA_DIR` | `data` | Face samples, trained model and SQLite database |
| `DATABASE_URL` | SQLite in the data dir | Any SQLAlchemy URL, e.g. `postgresql+psycopg://...` |
| `MATCH_THRESHOLD` | `70` | Max LBPH distance for a match; lower is stricter |
| `CONFIRM_FRAMES` | `5` | Consecutive frames needed before marking |

Tune `MATCH_THRESHOLD` on your own classroom: with `take`, the live view shows the distance next to each face.

## Database schema

- **students**: `roll_no` (unique), `name`, `class_name`
- **class_sessions**: `class_name`, `subject`, `started_at`, `ended_at`
- **attendance_records**: `session_id`, `student_id`, `marked_at`, `method` (face/manual), `confidence`, with a unique constraint on (`session_id`, `student_id`)

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

45 tests cover detection, recognition, enrollment, the confirmation logic, the live loop (with a fake camera), the service layer, the CLI and the full API flow. Real student photos can't be committed, so the tests use synthetic images: drawn faces that OpenCV's real Haar cascade detects, and per-student textured face crops for the recogniser. Validate recognition accuracy on real enrolled photos before relying on it. GitHub Actions runs lint and tests, then builds the Docker image and smoke-tests the API.

## Privacy

Face images and the trained model stay in the local data directory, which is git-ignored, and are never committed. Get consent before enrolling students, and delete a student's folder under `data/faces/` (then retrain) when they leave.

## Project structure

```
attendance/
├── face.py          # Haar detection + LBPH recognition
├── enroll.py        # storing face samples, training
├── recognition.py   # per-frame recognition + multi-frame confirmation
├── live.py          # webcam loop
├── models.py        # SQLAlchemy tables
├── service.py       # students, sessions, marking, reports
├── api.py           # FastAPI REST API
├── cli.py           # command-line interface
└── static/          # web dashboard
tests/               # pytest suite with synthetic images
```

## License

MIT
