import os

import numpy as np
import pandas as pd

from weaselytics.export import export_csv, export_dist, export_txt


class TestExportTxt:
    def test_creates_file(self, tmp_path):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.1, 0.5, 0.1])
        path = os.path.join(tmp_path, "data.txt")
        export_txt(x, y, path=path)
        expected = os.path.join(tmp_path, "data_bl.txt")
        assert os.path.exists(expected)

    def test_content_format(self, tmp_path):
        x = np.array([0.0, 1.0])
        y = np.array([0.1, 0.5])
        path = os.path.join(tmp_path, "data.txt")
        export_txt(x, y, path=path)
        expected = os.path.join(tmp_path, "data_bl.txt")
        content = np.loadtxt(expected)
        np.testing.assert_allclose(content[:, 0], x)
        np.testing.assert_allclose(content[:, 1], y)


class TestExportCsv:
    def test_creates_file(self, tmp_path):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.1, 0.5, 0.1])
        path = os.path.join(tmp_path, "data.txt")
        export_csv(x, y, path=path)
        expected = os.path.join(tmp_path, "data.csv")
        assert os.path.exists(expected)

    def test_content(self, tmp_path):
        x = np.array([0.0, 1.0])
        y = np.array([0.1, 0.5])
        path = os.path.join(tmp_path, "data.txt")
        export_csv(x, y, path=path)
        expected = os.path.join(tmp_path, "data.csv")
        df = pd.read_csv(expected)
        np.testing.assert_allclose(df["time"].values, x)
        np.testing.assert_allclose(df["potential"].values, y)


class TestExportDist:
    def test_creates_file(self, tmp_path):
        mol = "test_mol"
        g_fit = np.array([1.0, 5.0, 0.5])
        sn_fit = np.array([1.0, 5.0, 0.5, 0.0])
        path = os.path.join(tmp_path, "data__LPYE.txt")
        with open(path, "w") as f:
            f.write("dummy")
        export_dist(mol, g_fit, sn_fit, path)
        expected = os.path.join(tmp_path, "data__LPYE_test_mol.csv")
        assert os.path.exists(expected)
