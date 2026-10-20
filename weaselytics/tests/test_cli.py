import os

import pytest

from weaselytics.weaselytics import main


def test_missing_file(monkeypatch):
    with pytest.raises(FileNotFoundError):
        monkeypatch.setattr("sys.argv", ["weaselytics", "/nonexistent/file.txt"])
        main()


def test_equal_x0_x1(monkeypatch, sample_txt_path):
    monkeypatch.setattr("sys.argv", ["weaselytics", sample_txt_path, "-x0", "1.0", "-x1", "1.0"])
    with pytest.raises(ValueError, match="x0 and x1 are equal"):
        main()


def test_x0_greater_than_x1(monkeypatch, sample_txt_path):
    monkeypatch.setattr("sys.argv", ["weaselytics", sample_txt_path, "-x0", "3.0", "-x1", "1.0"])
    with pytest.raises(ValueError, match="x1 is larger than x0"):
        main()


def test_negative_x0(monkeypatch, sample_txt_path):
    monkeypatch.setattr("sys.argv", ["weaselytics", sample_txt_path, "-x0", "-1.0"])
    with pytest.raises(ValueError, match="x0 < 0"):
        main()


def test_negative_x1(monkeypatch, sample_txt_path):
    monkeypatch.setattr("sys.argv", ["weaselytics", sample_txt_path, "-x1", "-1.0"])
    with pytest.raises(ValueError, match="x1 < 0"):
        main()


def test_basic_invocation(monkeypatch, sample_txt_path):
    monkeypatch.setattr("sys.argv", ["weaselytics", sample_txt_path, "-nb", "-n"])
    main()


def test_output_csv_flag(monkeypatch, sample_txt_path, tmp_path):
    monkeypatch.setattr("sys.argv", [
        "weaselytics", sample_txt_path, "-nb", "-n",
        "-o", "-od", str(tmp_path),
    ])
    main()
    csv_files = list(tmp_path.rglob("*.csv"))
    assert len(csv_files) >= 1


def test_output_dir_is_used(monkeypatch, sample_txt_path, tmp_path):
    monkeypatch.setattr("sys.argv", [
        "weaselytics", sample_txt_path, "-nb", "-n",
        "-o", "-od", str(tmp_path),
    ])
    main()
    mobile_phase = os.path.basename(os.path.dirname(sample_txt_path))
    expected = tmp_path / mobile_phase / "sample_chromato.csv"
    assert expected.exists()


def test_smoothing_flag(monkeypatch, sample_txt_path, tmp_path):
    monkeypatch.setattr("sys.argv", [
        "weaselytics", sample_txt_path, "-nb", "-n",
        "-sm", "-o", "-od", str(tmp_path),
    ])
    main()
    mobile_phase = os.path.basename(os.path.dirname(sample_txt_path))
    expected = tmp_path / mobile_phase / "sample_chromato.csv"
    assert expected.exists()


def test_output_stats_flag(monkeypatch, sample_txt_path, tmp_path):
    monkeypatch.setattr("sys.argv", [
        "weaselytics", sample_txt_path, "-nb",
        "-os", "test_mol", "-od", str(tmp_path),
    ])
    main()
    mobile_phase = os.path.basename(os.path.dirname(sample_txt_path))
    expected = tmp_path / mobile_phase / "sample_chromato_test_mol.csv"
    assert expected.exists()


def _multipeak_txt(tmp_path):
    """Chromatogram-like signal accepted by ``_relevant_regions`` (the
    single-peak sample fixture is filtered out by its peak-relevance
    criteria, so the custom_beads CLI path cannot run on it)."""
    import numpy as np

    def gauss(x, a, mu, sig):
        return a * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    x = np.linspace(0, 20, 1200)
    y = (gauss(x, 5, 4, 0.15) + gauss(x, 8, 7, 0.2)
         + gauss(x, 4, 11, 0.25) + 0.3 + 0.05 * x
         + 0.02 * np.random.default_rng(0).normal(size=len(x)))
    path = tmp_path / "multipeak.txt"
    lines = [f"{xv:.6f}\t{yv:.6f}" for xv, yv in zip(x, y)]
    path.write_text("\n".join(lines))
    return str(path)


def test_freq_cutoff_flag_bypasses_selection(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("sys.argv", [
        "weaselytics", _multipeak_txt(tmp_path), "-n", "-fc", "0.01",
        "-od", str(tmp_path),
    ])
    main()
    out = capsys.readouterr().out
    # No autocorrelation sweep was run, the given value was used as-is
    assert "Autocorrelation in" not in out
    assert "1.0000E-02" in out


def test_freq_cutoff_flag_with_print_plot(monkeypatch, tmp_path):
    # A fixed cutoff produces no r2 diagnostic data; -p must not crash
    monkeypatch.setattr("sys.argv", [
        "weaselytics", _multipeak_txt(tmp_path), "-n", "-fc", "0.01",
        "-p", "-od", str(tmp_path),
    ])
    main()
    assert not (tmp_path / "r2_plots").exists()


def test_invalid_freq_cutoff_flag(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", [
        "weaselytics", _multipeak_txt(tmp_path), "-n", "-fc", "0.7",
    ])
    with pytest.raises(ValueError, match="cutoff frequency"):
        main()
