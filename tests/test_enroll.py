import numpy as np
import pytest

from attendance.enroll import enroll_image, load_samples, sample_counts, student_dir, train_from_disk
from attendance.face import FaceDetector
from tests.synthetic import draw_face, identity_face
from attendance.enroll import save_face


@pytest.fixture(scope="module")
def detector():
    return FaceDetector(min_size=40)


def test_enroll_saves_one_crop_per_image(detector, tmp_path):
    for k in range(3):
        enroll_image(draw_face(k), "CS101", str(tmp_path), detector)
    assert sample_counts(str(tmp_path)) == {"CS101": 3}
    assert sorted(p.name for p in (tmp_path / "CS101").iterdir()) == ["0001.png", "0002.png", "0003.png"]


def test_enroll_rejects_images_without_exactly_one_face(detector, tmp_path):
    with pytest.raises(ValueError, match="found 0"):
        enroll_image(np.zeros((300, 300, 3), np.uint8), "CS101", str(tmp_path), detector)
    with pytest.raises(ValueError, match="found 2"):
        enroll_image(np.hstack([draw_face(1), draw_face(2)]), "CS101", str(tmp_path), detector)


@pytest.mark.parametrize("roll", ["../evil", "a/b", "", "x" * 40, "name with space"])
def test_unsafe_roll_numbers_are_rejected(roll, tmp_path):
    with pytest.raises(ValueError):
        student_dir(str(tmp_path), roll)


def test_train_from_disk(tmp_path):
    faces = tmp_path / "faces"
    for s in range(4):
        for k in range(4):
            save_face(str(faces), f"R{s}", identity_face(s, k))
    recognizer, n_students, n_faces = train_from_disk(str(faces), str(tmp_path / "m.yml"), threshold=40)
    assert (n_students, n_faces) == (4, 16)
    assert recognizer.predict(identity_face(2, 50))[0] == "R2"
    assert (tmp_path / "m.yml").exists() and (tmp_path / "m.yml.labels.json").exists()
    assert set(load_samples(str(faces))) == {"R0", "R1", "R2", "R3"}
