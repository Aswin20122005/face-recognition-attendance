"""Synthetic test images (real student photos can't be committed).

- draw_face(): a simple drawn face that OpenCV's frontal-face Haar cascade detects,
  with small random shifts, lighting changes and noise. Used to test detection.
- identity_face(): a 200x200 preprocessed "face crop" whose texture is unique per
  student, plus per-sample noise and lighting. Used to test recognition, which
  only ever sees preprocessed crops.
"""

import cv2
import numpy as np


def draw_face(sample=0, size=300, center=None):
    rng = np.random.default_rng(sample)
    cx, cy = center or (size // 2, size // 2)
    cx, cy = cx + int(rng.integers(-5, 6)), cy + int(rng.integers(-5, 6))
    img = np.full((size, size), 200, np.uint8)
    cv2.ellipse(img, (cx, cy), (80, 105), 0, 0, 360, 150, -1)
    for sx in (-1, 1):
        ex = cx + sx * 32
        cv2.ellipse(img, (ex, cy - 25), (16, 8), 0, 0, 360, 60, -1)
        cv2.circle(img, (ex, cy - 25), 5, 20, -1)
        cv2.line(img, (ex - 18, cy - 45), (ex + 18, cy - 47), 60, 5)
    cv2.line(img, (cx, cy - 15), (cx - 5, cy + 25), 110, 4)
    cv2.ellipse(img, (cx, cy + 50), (28, 10), 0, 0, 180, 70, 4)
    img = cv2.GaussianBlur(img, (7, 7), 0).astype(np.float32)
    img = img * rng.uniform(0.9, 1.1) + rng.normal(0, 3, img.shape)
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def identity_face(student_id, sample=0, size=200):
    base_rng = np.random.default_rng(1000 + student_id)
    texture = base_rng.integers(0, 256, (size // 8, size // 8)).astype(np.uint8)
    face = cv2.resize(texture, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    rng = np.random.default_rng(student_id * 1000 + sample + 1)
    face = face * rng.uniform(0.8, 1.2) + rng.normal(0, 8, face.shape)
    shift = rng.integers(-3, 4, 2)
    face = np.roll(np.clip(face, 0, 255).astype(np.uint8), tuple(int(v) for v in shift), axis=(0, 1))
    return cv2.equalizeHist(face)
