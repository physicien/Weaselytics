import numpy as np

from weaselytics.parsers import ParsedData


class TestParsedData:
    def test_reads_sample_file(self, sample_txt_path):
        parsed = ParsedData(sample_txt_path)
        assert parsed.data is not None
        assert parsed.data.shape == (2, 21)

    def test_x_values(self, sample_txt_path):
        parsed = ParsedData(sample_txt_path)
        x = parsed.data[0]
        expected = np.linspace(0, 2.0, 21)
        np.testing.assert_allclose(x, expected)

    def test_y_values(self, sample_txt_path):
        parsed = ParsedData(sample_txt_path)
        y = parsed.data[1]
        expected = np.array([
            0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.2, 3.8,
            4.0, 3.8, 3.2, 2.5, 1.8, 1.2, 0.8, 0.5, 0.3, 0.2, 0.1
        ])
        np.testing.assert_allclose(y, expected)

    def test_raises_on_nonexistent_file(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            ParsedData("/nonexistent/file.txt")

    def test_integer_data_lines_are_kept(self, tmp_path):
        # Acquisitions start with integer time stamps (0, 1, 2, ...);
        # these lines were silently dropped by the previous pattern
        path = tmp_path / "integers.txt"
        path.write_text("header line\n0\t0.5\n1\t-0.25\n2.5\t3\n3.5\t4.5\n")
        parsed = ParsedData(str(path))
        assert parsed.data.shape == (2, 4)
        np.testing.assert_allclose(parsed.data[0], [0.0, 1.0, 2.5, 3.5])

    def test_scientific_notation(self, tmp_path):
        path = tmp_path / "scientific.txt"
        path.write_text("1.5e-3\t-2E+2\n2e3\t0\n")
        parsed = ParsedData(str(path))
        np.testing.assert_allclose(parsed.data[0], [1.5e-3, 2e3])
        np.testing.assert_allclose(parsed.data[1], [-200.0, 0.0])

    def test_malformed_exponent_is_skipped(self, tmp_path):
        # '1.5e' is not a number; the line must be skipped, not crash
        path = tmp_path / "malformed.txt"
        path.write_text("1.5e\t2.0\n1.0\t2.0\n")
        parsed = ParsedData(str(path))
        assert parsed.data.shape == (2, 1)
