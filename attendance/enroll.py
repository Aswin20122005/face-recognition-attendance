"""Store students' face samples on disk and train the recognizer from them.

Layout: <data>/faces/<roll_no>/0001.png, 0002.png, ...  (preprocessed crops)
"""

from __future__ import annotations

import glob
import os
import re

import cv2

from attendance.face import FaceDetector, FaceRecognizer

SAFE_ROLL = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def student_dir(faces_dir, roll_no):
    # Roll numbers become folder names, so reject anything that could escape faces_dir.
    if not SAFE_ROLL.match(roll_no):
        raise ValueError(f"Invalid roll number {roll_no!r}: use letters, digits, '-' or '_'")
    return os.path.join(faces_dir, roll_no)


def save_face(faces_dir, roll_no, face):
    folder = student_dir(faces_dir, roll_no)
    os.makedirs(folder, exist_ok=True)
    n = len(glob.glob(os.path.join(folder, "*.png"))) + 1
    path = os.path.join(folder, f"{n:04d}.png")
    cv2.imwrite(path, face)
    return path


def enroll_image(image, roll_no, faces_dir, detector=None):
    """Detect exactly one face in an image and store it. Returns the saved path.

    Images with zero or several faces are rejected: with a group photo we can't
    know which face belongs to the student.
    """
    detector = detector or FaceDetector()
    found = detector.detect(image)
    if len(found) != 1:
        raise ValueError(f"Expected exactly one face, found {len(found)}")
    return save_face(faces_dir, roll_no, found[0].face)


def load_samples(faces_dir):
    """dict roll_no -> list of face images, read from the faces folder."""
    samples = {}
    for folder in sorted(glob.glob(os.path.join(faces_dir, "*"))):
        if not os.path.isdir(folder):
            continue
        images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in sorted(glob.glob(os.path.join(folder, "*.png")))]
        images = [img for img in images if img is not None]
        if images:
            samples[os.path.basename(folder)] = images
    return samples


def sample_counts(faces_dir):
    return {roll: len(imgs) for roll, imgs in load_samples(faces_dir).items()}


def train_from_disk(faces_dir, model_path, threshold=70.0):
    """Train on every enrolled face and save the model. Returns (recognizer, n_students, n_faces)."""
    samples = load_samples(faces_dir)
    recognizer = FaceRecognizer(threshold=threshold)
    n_faces = recognizer.train(samples)
    recognizer.save(model_path)
    return recognizer, len(samples), n_faces
