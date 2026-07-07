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

    def test_stores_path(self, sample_txt_path):
        parsed = ParsedData(sample_txt_path)
        assert parsed.path == sample_txt_path

    def test_raises_on_nonexistent_file(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            ParsedData("/nonexistent/file.txt")
