"""Settings, overridable with environment variables."""

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    data_dir: str = field(default_factory=lambda: os.environ.get("ATTENDANCE_DATA_DIR", "data"))
    database_url: str = field(default_factory=lambda: os.environ.get("DATABASE_URL", ""))
    # LBPH distance: lower = more similar. Faces above this are reported as unknown.
    match_threshold: float = field(default_factory=lambda: float(os.environ.get("MATCH_THRESHOLD", "70")))
    # A student must be recognised in this many consecutive frames before being marked.
    confirm_frames: int = field(default_factory=lambda: int(os.environ.get("CONFIRM_FRAMES", "5")))
    face_size: int = 200  # faces are resized to face_size x face_size before recognition

    @property
    def faces_dir(self):
        return os.path.join(self.data_dir, "faces")

    @property
    def model_path(self):
        return os.path.join(self.data_dir, "recognizer.yml")

    @property
    def db_url(self):
        return self.database_url or f"sqlite:///{os.path.join(self.data_dir, 'attendance.db')}"


def get_settings():
    return Settings()
