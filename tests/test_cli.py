import cv2

from attendance import cli
from tests.synthetic import draw_face


def run(monkeypatch, tmp_path, *args):
    monkeypatch.setenv("ATTENDANCE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MATCH_THRESHOLD", "80")
    return cli.main(list(args))


def test_add_enroll_train_and_report(monkeypatch, tmp_path, capsys):
    photos = tmp_path / "photos"
    photos.mkdir()
    for k in range(4):
        cv2.imwrite(str(photos / f"{k}.png"), draw_face(k))
    cv2.imwrite(str(photos / "empty.png"), draw_face(0)[:40, :40])

    assert run(monkeypatch, tmp_path, "add-student", "CS001", "Asha", "--class", "CSE-A") == 0
    assert run(monkeypatch, tmp_path, "enroll", "CS001", "--images", str(photos)) == 0
    out = capsys.readouterr().out
    assert "Saved 4 face samples" in out and "Skipped" in out

    assert run(monkeypatch, tmp_path, "train") == 0
    assert "Trained on 4 faces of 1 students" in capsys.readouterr().out

    csv_path = tmp_path / "summary.csv"
    assert run(monkeypatch, tmp_path, "report", "--class", "CSE-A", "--csv", str(csv_path)) == 0
    assert csv_path.read_text().startswith("roll_no,name,attended,sessions,percentage")


def test_errors_return_exit_code_1(monkeypatch, tmp_path, capsys):
    assert run(monkeypatch, tmp_path, "enroll", "NOPE", "--images", str(tmp_path)) == 1
    assert "not found" in capsys.readouterr().err
    run(monkeypatch, tmp_path, "add-student", "CS001", "A")
    assert run(monkeypatch, tmp_path, "add-student", "CS001", "B") == 1
