import numpy as np
import pytest

from attendance.face import FaceDetector, FaceRecognizer, preprocess
from tests.synthetic import draw_face, identity_face

DETECTED_SAMPLES = [0, 1, 2, 3, 4, 5]  # drawn faces the cascade finds (deterministic seeds)


@pytest.fixture(scope="module")
def detector():
    return FaceDetector(min_size=40)


def test_no_face_in_blank_image(detector):
    assert detector.detect(np.zeros((480, 640, 3), np.uint8)) == []


@pytest.mark.parametrize("sample", DETECTED_SAMPLES)
def test_detects_one_face_and_returns_preprocessed_crop(detector, sample):
    [det] = detector.detect(draw_face(sample))
    x, y, w, h = det.box
    assert w > 100 and h > 100
    assert det.face.shape == (200, 200) and det.face.dtype == np.uint8


def test_detects_two_faces_side_by_side(detector):
    assert len(detector.detect(np.hstack([draw_face(1), draw_face(2)]))) == 2


def test_preprocess_equalises_and_resizes():
    out = preprocess(np.full((50, 80, 3), 120, np.uint8), size=64)
    assert out.shape == (64, 64)


@pytest.fixture(scope="module")
def recognizer():
    r = FaceRecognizer(threshold=40)
    r.train({f"R{s:03d}": [identity_face(s, k) for k in range(5)] for s in range(8)})
    return r


def test_recognises_new_samples_of_enrolled_students(recognizer):
    for s in range(8):
        for k in range(10, 13):
            roll, distance = recognizer.predict(identity_face(s, k))
            assert roll == f"R{s:03d}", (s, k, distance)


def test_unknown_person_is_rejected(recognizer):
    roll, distance = recognizer.predict(identity_face(99, 0))
    assert roll is None and distance > 40


def test_save_and_load_round_trip(recognizer, tmp_path):
    path = str(tmp_path / "model.yml")
    recognizer.save(path)
    loaded = FaceRecognizer(threshold=40).load(path)
    assert loaded.predict(identity_face(3, 20))[0] == "R003"


def test_untrained_recognizer_refuses_to_predict():
    with pytest.raises(RuntimeError):
        FaceRecognizer().predict(identity_face(0))


def test_training_needs_samples():
    with pytest.raises(ValueError):
        FaceRecognizer().train({})
