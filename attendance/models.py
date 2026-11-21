"""Tables: students, class sessions and attendance records."""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from attendance.db import Base


def utcnow():
    return datetime.now(timezone.utc)


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(primary_key=True)
    roll_no: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    class_name: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    records: Mapped[list["AttendanceRecord"]] = relationship(back_populates="student", cascade="all, delete-orphan")


class ClassSession(Base):
    """One lecture / period for which attendance is taken."""

    __tablename__ = "class_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    class_name: Mapped[str] = mapped_column(String(64))
    subject: Mapped[str] = mapped_column(String(120), default="")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    records: Mapped[list["AttendanceRecord"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"
    # A student can only be marked present once per session.
    __table_args__ = (UniqueConstraint("session_id", "student_id", name="uq_session_student"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("class_sessions.id"))
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"))
    marked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    method: Mapped[str] = mapped_column(String(16), default="face")  # face | manual
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)  # LBPH distance

    session: Mapped[ClassSession] = relationship(back_populates="records")
    student: Mapped[Student] = relationship(back_populates="records")
