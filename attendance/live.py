"""Webcam attendance: recognise students in real time and mark them present."""

from __future__ import annotations

import cv2

from attendance import service
from attendance.recognition import ConfirmationTracker, process_frame

GREEN, ORANGE, RED = (0, 200, 0), (0, 165, 255), (0, 0, 255)


def draw_results(frame, results, names, marked):
    for r in results:
        x, y, w, h = r.box
        if r.roll_no is None:
            color, label = RED, f"Unknown ({r.distance:.0f})"
        elif r.roll_no in marked:
            color, label = GREEN, f"{names.get(r.roll_no, r.roll_no)} - present"
        else:
            color, label = ORANGE, f"{names.get(r.roll_no, r.roll_no)} ({r.distance:.0f})"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, max(y - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Present: {len(marked)}   q: finish", (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def run_live(db, session_id, detector, recognizer, camera=0, confirm_frames=5, show=True, max_frames=None,
             capture=None):
    """Run until 'q' is pressed (or max_frames). Returns the roll numbers marked present."""
    names = {s.roll_no: s.name for s in service.list_students(db)}
    tracker = ConfirmationTracker(required=confirm_frames)
    marked = {row["roll_no"] for row in service.session_report(db, session_id)["students"]
              if row["status"] == "present"}
    tracker.confirmed |= marked  # already present (e.g. after a restart) won't be re-confirmed
    cap = capture or cv2.VideoCapture(camera)
    frames = 0
    try:
        while max_frames is None or frames < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera returned no frame; stopping.")
                break
            frames += 1
            results = process_frame(frame, detector, recognizer, tracker)
            for r in results:
                if r.confirmed:
                    _, created = service.mark_present(db, session_id, r.roll_no, confidence=r.distance)
                    if created:
                        print(f"Marked present: {r.roll_no} {names.get(r.roll_no, '')} (distance {r.distance:.1f})")
                    marked.add(r.roll_no)
            if show:
                cv2.imshow("Attendance", draw_results(frame, results, names, marked))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
    return marked
