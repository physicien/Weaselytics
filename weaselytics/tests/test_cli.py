import os
import sys

import pytest

from weaselytics.weaselytics import main


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        sys.argv = ["weaselytics", "/nonexistent/file.txt"]
        main()


def test_equal_x0_x1(sample_txt_path):
    sys.argv = ["weaselytics", sample_txt_path, "-x0", "1.0", "-x1", "1.0"]
    with pytest.raises(ValueError, match="x0 and x1 are equal"):
        main()


def test_x0_greater_than_x1(sample_txt_path):
    sys.argv = ["weaselytics", sample_txt_path, "-x0", "3.0", "-x1", "1.0"]
    with pytest.raises(ValueError, match="x1 is larger than x0"):
        main()


def test_negative_x0(sample_txt_path):
    sys.argv = ["weaselytics", sample_txt_path, "-x0", "-1.0"]
    with pytest.raises(ValueError, match="x0 < 0"):
        main()


def test_negative_x1(sample_txt_path):
    sys.argv = ["weaselytics", sample_txt_path, "-x1", "-1.0"]
    with pytest.raises(ValueError, match="x1 < 0"):
        main()


def test_basic_invocation(sample_txt_path):
    sys.argv = ["weaselytics", sample_txt_path, "-nb", "-n"]
    main()


def test_output_csv_flag(sample_txt_path, tmp_path):
    sys.argv = [
        "weaselytics", sample_txt_path, "-nb", "-n",
        "-o", "-od", str(tmp_path),
    ]
    main()
    csv_files = list(tmp_path.rglob("*.csv"))
    assert len(csv_files) >= 1


def test_output_dir_is_used(sample_txt_path, tmp_path):
    sys.argv = [
        "weaselytics", sample_txt_path, "-nb", "-n",
        "-o", "-od", str(tmp_path),
    ]
    main()
    mobile_phase = os.path.basename(os.path.dirname(sample_txt_path))
    expected = tmp_path / mobile_phase / "sample_chromato.csv"
    assert expected.exists()


def test_smoothing_flag(sample_txt_path, tmp_path):
    sys.argv = [
        "weaselytics", sample_txt_path, "-nb", "-n",
        "-sm", "-o", "-od", str(tmp_path),
    ]
    main()
    mobile_phase = os.path.basename(os.path.dirname(sample_txt_path))
    expected = tmp_path / mobile_phase / "sample_chromato.csv"
    assert expected.exists()


def test_output_stats_flag(sample_txt_path, tmp_path):
    sys.argv = [
        "weaselytics", sample_txt_path, "-nb",
        "-os", "test_mol", "-od", str(tmp_path),
    ]
    main()
    mobile_phase = os.path.basename(os.path.dirname(sample_txt_path))
    expected = tmp_path / mobile_phase / "sample_chromato_test_mol.csv"
    assert expected.exists()
