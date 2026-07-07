import sys

import pytest

from weaselytics.weaselytics import main


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        sys.argv = ["weaselytics", "/nonexistent/file.txt"]
        main()


def test_equal_x0_x1(sample_txt_path, capsys):
    sys.argv = ["weaselytics", sample_txt_path, "-x0", "1.0", "-x1", "1.0"]
    with pytest.raises(SystemExit):
        main()
    captured = capsys.readouterr()
    assert "x0 and x1 are equal" in captured.out


def test_x0_greater_than_x1(sample_txt_path, capsys):
    sys.argv = ["weaselytics", sample_txt_path, "-x0", "3.0", "-x1", "1.0"]
    with pytest.raises(SystemExit):
        main()
    captured = capsys.readouterr()
    assert "x1 is larger than x0" in captured.out


def test_negative_x0(sample_txt_path, capsys):
    sys.argv = ["weaselytics", sample_txt_path, "-x0", "-1.0"]
    with pytest.raises(SystemExit):
        main()
    captured = capsys.readouterr()
    assert "x0 < 0" in captured.out


def test_negative_x1(sample_txt_path, capsys):
    sys.argv = ["weaselytics", sample_txt_path, "-x1", "-1.0"]
    with pytest.raises(SystemExit):
        main()
    captured = capsys.readouterr()
    assert "x1 < 0" in captured.out


def test_basic_invocation(sample_txt_path):
    sys.argv = ["weaselytics", sample_txt_path, "-nb", "-n"]
    main()
