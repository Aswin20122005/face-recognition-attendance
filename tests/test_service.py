import pytest

from attendance import service
from attendance.service import Conflict, NotFound


def setup_class(db, n=3):
    for i in range(n):
        service.add_student(db, f"CS{i:03d}", f"Student {i}", "CSE-A")
    service.add_student(db, "EC001", "Other class", "ECE-B")


def test_add_and_list_students_by_class(db):
    setup_class(db)
    assert [s.roll_no for s in service.list_students(db, "CSE-A")] == ["CS000", "CS001", "CS002"]
    assert len(service.list_students(db)) == 4


def test_duplicate_roll_number_is_rejected(db):
    service.add_student(db, "CS000", "A")
    with pytest.raises(Conflict):
        service.add_student(db, "CS000", "B")


def test_marking_is_idempotent(db):
    setup_class(db)
    s = service.start_session(db, "CSE-A", "DBMS")
    rec, created = service.mark_present(db, s.id, "CS001", confidence=31.5)
    again, created_again = service.mark_present(db, s.id, "CS001")
    assert created and not created_again
    assert rec.id == again.id and rec.confidence == 31.5


def test_unknown_student_or_session_raise_not_found(db):
    setup_class(db)
    s = service.start_session(db, "CSE-A")
    with pytest.raises(NotFound):
        service.mark_present(db, s.id, "NOPE")
    with pytest.raises(NotFound):
        service.mark_present(db, 999, "CS000")


def test_cannot_mark_after_session_ends(db):
    setup_class(db)
    s = service.start_session(db, "CSE-A")
    service.end_session(db, s.id)
    with pytest.raises(Conflict):
        service.mark_present(db, s.id, "CS000")


def test_session_report_lists_present_and_absent_from_the_roster(db):
    setup_class(db)
    s = service.start_session(db, "CSE-A", "DBMS")
    service.mark_present(db, s.id, "CS000")
    service.mark_present(db, s.id, "CS002", method="manual")
    report = service.session_report(db, s.id)
    assert (report["present"], report["absent"], report["total"]) == (2, 1, 3)
    status = {r["roll_no"]: r["status"] for r in report["students"]}
    assert status == {"CS000": "present", "CS001": "absent", "CS002": "present"}
    assert "EC001" not in status  # other class not on the roster


def test_student_summary_percentages(db):
    setup_class(db)
    sessions = [service.start_session(db, "CSE-A") for _ in range(4)]
    for s in sessions[:3]:
        service.mark_present(db, s.id, "CS000")
    service.mark_present(db, sessions[0].id, "CS001")
    summary = {r["roll_no"]: r for r in service.student_summary(db, "CSE-A")}
    assert summary["CS000"]["percentage"] == 75.0
    assert summary["CS001"]["percentage"] == 25.0
    assert summary["CS002"]["attended"] == 0 and summary["CS002"]["sessions"] == 4


def test_deleting_a_student_removes_their_records(db):
    setup_class(db)
    s = service.start_session(db, "CSE-A")
    service.mark_present(db, s.id, "CS000")
    service.delete_student(db, "CS000")
    assert service.session_report(db, s.id)["total"] == 2
