"""Face detection (Haar cascade) and recognition (LBPH), both built into OpenCV.

LBPH (Local Binary Patterns Histograms) is a good fit for a classroom kiosk: it
trains in seconds on a CPU, needs only a handful of photos per student, can be
retrained incrementally, and copes with moderate lighting changes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x, y, w, h
    face: np.ndarray                 # preprocessed grayscale crop


class FaceDetector:
    def __init__(self, face_size=200, min_size=60):
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {path}")
        self.face_size = face_size
        self.min_size = min_size

    def detect(self, image):
        """Return a Detection for every face found in a BGR or grayscale image."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_size, self.min_size)
        )
        return [Detection(tuple(int(v) for v in b), preprocess(gray[b[1]:b[1] + b[3], b[0]:b[0] + b[2]],
                                                               self.face_size))
                for b in boxes]


def preprocess(face, size=200):
    """Grayscale, fixed size and histogram-equalised, so lighting matters less."""
    if face.ndim == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(face)


class FaceRecognizer:
    """LBPH recognizer that maps OpenCV's integer labels to student roll numbers."""

    def __init__(self, threshold=70.0):
        self.threshold = threshold
        self.model = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        self.labels: dict[int, str] = {}
        self.trained = False

    def train(self, samples):
        """samples: dict roll_no -> list of preprocessed face images."""
        faces, ids = [], []
        self.labels = {}
        for i, (roll_no, images) in enumerate(sorted(samples.items())):
            self.labels[i] = roll_no
            for img in images:
                faces.append(img)
                ids.append(i)
        if not faces:
            raise ValueError("No face samples to train on")
        self.model.train(faces, np.array(ids, dtype=np.int32))
        self.trained = True
        return len(faces)

    def predict(self, face):
        """Return (roll_no or None, distance). None means 'unknown face'."""
        if not self.trained:
            raise RuntimeError("Recognizer has not been trained")
        label, distance = self.model.predict(face)
        if distance > self.threshold:
            return None, float(distance)
        return self.labels.get(int(label)), float(distance)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.write(path)
        with open(path + ".labels.json", "w") as f:
            json.dump({str(k): v for k, v in self.labels.items()}, f)

    def load(self, path):
        self.model.read(path)
        with open(path + ".labels.json") as f:
            self.labels = {int(k): v for k, v in json.load(f).items()}
        self.trained = True
        return self
