"""Turn camera frames into attendance marks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FaceResult:
    box: tuple[int, int, int, int]
    roll_no: str | None   # None = unknown face
    distance: float
    confirmed: bool       # True on the frame the student reached the confirmation count


class ConfirmationTracker:
    """Only confirm a student after `required` consecutive frames.

    One misrecognised frame (motion blur, a half-turned head) would otherwise be
    enough to mark the wrong student present. A student who leaves the frame starts
    from zero again; each student is confirmed at most once.
    """

    def __init__(self, required=5):
        self.required = required
        self.streaks: dict[str, int] = {}
        self.confirmed: set[str] = set()

    def update(self, rolls_in_frame):
        """Feed the roll numbers recognised in this frame; returns the newly confirmed ones."""
        seen = set(r for r in rolls_in_frame if r is not None)
        self.streaks = {r: self.streaks.get(r, 0) + 1 for r in seen}
        newly = {r for r, n in self.streaks.items() if n >= self.required and r not in self.confirmed}
        self.confirmed |= newly
        return newly


def process_frame(frame, detector, recognizer, tracker):
    """Detect and recognise every face in a frame. Returns a list of FaceResult."""
    detections = detector.detect(frame)
    predictions = [recognizer.predict(d.face) for d in detections]
    newly = tracker.update([roll for roll, _ in predictions])
    return [
        FaceResult(d.box, roll, dist, roll in newly)
        for d, (roll, dist) in zip(detections, predictions)
    ]
