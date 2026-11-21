from attendance.face import Detection, FaceRecognizer
from attendance.recognition import ConfirmationTracker, process_frame
from tests.synthetic import identity_face


def test_student_confirmed_after_required_consecutive_frames():
    t = ConfirmationTracker(required=3)
    assert t.update(["A"]) == set()
    assert t.update(["A"]) == set()
    assert t.update(["A"]) == {"A"}
    assert t.update(["A"]) == set()  # only once


def test_streak_resets_when_student_leaves_the_frame():
    t = ConfirmationTracker(required=3)
    t.update(["A"])
    t.update(["A"])
    t.update([])
    assert t.update(["A"]) == set()


def test_unknown_faces_never_confirm_and_several_students_count_independently():
    t = ConfirmationTracker(required=2)
    t.update([None, "A", "B"])
    assert t.update([None, "A", "B"]) == {"A", "B"}


class FakeDetector:
    """Returns pre-made face crops, so the pipeline can be tested without drawing scenes."""

    def __init__(self, faces):
        self.faces = faces

    def detect(self, frame):
        return [Detection((i * 50, 0, 40, 40), f) for i, f in enumerate(self.faces)]


def test_process_frame_recognises_and_confirms():
    rec = FaceRecognizer(threshold=40)
    rec.train({"R1": [identity_face(1, k) for k in range(4)], "R2": [identity_face(2, k) for k in range(4)]})
    tracker = ConfirmationTracker(required=2)
    detector = FakeDetector([identity_face(1, 20), identity_face(99, 0)])

    first = process_frame(None, detector, rec, tracker)
    second = process_frame(None, detector, rec, tracker)
    assert [r.roll_no for r in first] == ["R1", None]
    assert [r.confirmed for r in first] == [False, False]
    assert [r.confirmed for r in second] == [True, False]


class FakeCapture:
    def __init__(self, n):
        self.n = n

    def read(self):
        if self.n == 0:
            return False, None
        self.n -= 1
        return True, object()

    def release(self):
        pass


def test_run_live_marks_confirmed_students_once(db):
    from attendance import service
    from attendance.live import run_live

    service.add_student(db, "R1", "Asha", "CSE-A")
    service.add_student(db, "R2", "Ravi", "CSE-A")
    session = service.start_session(db, "CSE-A")
    rec = FaceRecognizer(threshold=40)
    rec.train({"R1": [identity_face(1, k) for k in range(4)], "R2": [identity_face(2, k) for k in range(4)]})
    detector = FakeDetector([identity_face(1, 30)])

    marked = run_live(db, session.id, detector, rec, confirm_frames=3, show=False, capture=FakeCapture(10))
    assert marked == {"R1"}
    report = service.session_report(db, session.id)
    assert report["present"] == 1 and report["absent"] == 1
