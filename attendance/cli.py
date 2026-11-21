"""Command-line interface.

    python -m attendance.cli add-student CS001 "Asha Kumar" --class CSE-A
    python -m attendance.cli enroll CS001 --camera 0 --samples 20    # or --images photos/asha/
    python -m attendance.cli train
    python -m attendance.cli take --class CSE-A --subject DBMS        # webcam attendance
    python -m attendance.cli report --session 1                       # or --class CSE-A --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import cv2

from attendance import service
from attendance.config import get_settings
from attendance.db import init_db, make_engine, make_session_factory
from attendance.enroll import enroll_image, sample_counts, train_from_disk
from attendance.face import FaceDetector, FaceRecognizer


def open_db(settings):
    engine = make_engine(settings.db_url)
    init_db(engine)
    return make_session_factory(engine)()


def cmd_add_student(args, settings):
    db = open_db(settings)
    s = service.add_student(db, args.roll_no, args.name, args.class_name)
    print(f"Added {s.roll_no} {s.name} ({s.class_name or 'no class'})")


def cmd_enroll(args, settings):
    db = open_db(settings)
    service.get_student(db, args.roll_no)  # must exist first
    detector = FaceDetector(face_size=settings.face_size)
    saved = 0
    if args.images:
        paths = sorted(p for ext in ("jpg", "jpeg", "png") for p in glob.glob(os.path.join(args.images, f"*.{ext}")))
        for path in paths:
            try:
                enroll_image(cv2.imread(path), args.roll_no, settings.faces_dir, detector)
                saved += 1
            except ValueError as e:
                print(f"Skipped {path}: {e}")
    else:
        cap = cv2.VideoCapture(args.camera)
        print("Look at the camera; move your head slightly between captures. Press q to stop.")
        while saved < args.samples:
            ok, frame = cap.read()
            if not ok:
                break
            found = detector.detect(frame)
            if len(found) == 1:
                enroll_image(frame, args.roll_no, settings.faces_dir, detector)
                saved += 1
                x, y, w, h = found[0].box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(frame, f"{saved}/{args.samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
            cv2.imshow("Enroll", frame)
            if cv2.waitKey(200) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    print(f"Saved {saved} face samples for {args.roll_no}")


def cmd_train(args, settings):
    _, n_students, n_faces = train_from_disk(settings.faces_dir, settings.model_path, settings.match_threshold)
    print(f"Trained on {n_faces} faces of {n_students} students -> {settings.model_path}")
    for roll, n in sorted(sample_counts(settings.faces_dir).items()):
        if n < 5:
            print(f"  warning: {roll} has only {n} samples; 10-20 give better accuracy")


def cmd_take(args, settings):
    from attendance.live import run_live

    db = open_db(settings)
    recognizer = FaceRecognizer(settings.match_threshold).load(settings.model_path)
    session = service.start_session(db, args.class_name, args.subject)
    print(f"Session {session.id}: {args.class_name} {args.subject}".strip())
    try:
        run_live(db, session.id, FaceDetector(face_size=settings.face_size), recognizer,
                 camera=args.camera, confirm_frames=settings.confirm_frames)
    finally:
        service.end_session(db, session.id)
    print_report(service.session_report(db, session.id))


def print_report(report):
    print(f"\nSession {report['session_id']} · {report['class_name']} {report['subject']}".rstrip())
    print(f"Present {report['present']} / {report['total']}")
    for row in report["students"]:
        print(f"  {row['roll_no']:<10} {row['name']:<25} {row['status']}")


def cmd_report(args, settings):
    db = open_db(settings)
    if args.session:
        report = service.session_report(db, args.session)
        print_report(report)
        rows = report["students"]
    else:
        rows = service.student_summary(db, args.class_name)
        for r in rows:
            print(f"  {r['roll_no']:<10} {r['name']:<25} {r['attended']}/{r['sessions']}  {r['percentage']}%")
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")


def build_parser():
    p = argparse.ArgumentParser(prog="attendance", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("add-student")
    a.add_argument("roll_no")
    a.add_argument("name")
    a.add_argument("--class", dest="class_name", default="")
    a.set_defaults(func=cmd_add_student)

    e = sub.add_parser("enroll")
    e.add_argument("roll_no")
    e.add_argument("--images", help="folder of photos (one face each) instead of the webcam")
    e.add_argument("--camera", type=int, default=0)
    e.add_argument("--samples", type=int, default=20)
    e.set_defaults(func=cmd_enroll)

    t = sub.add_parser("train")
    t.set_defaults(func=cmd_train)

    k = sub.add_parser("take", help="take attendance from the webcam")
    k.add_argument("--class", dest="class_name", required=True)
    k.add_argument("--subject", default="")
    k.add_argument("--camera", type=int, default=0)
    k.set_defaults(func=cmd_take)

    r = sub.add_parser("report")
    group = r.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", type=int)
    group.add_argument("--class", dest="class_name")
    r.add_argument("--csv", help="also write the report to a CSV file")
    r.set_defaults(func=cmd_report)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args, get_settings())
    except (service.NotFound, service.Conflict, ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
