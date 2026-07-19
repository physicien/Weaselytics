import numpy as np

from weaselytics.plot import plot, r2_plots


class TestPlot:
    def test_plot_basic(self, sample_data):
        x, y = sample_data
        plot(x, y)

    def test_plot_with_fits(self, sample_data):
        x, y = sample_data
        plot(x, y, y_sm=y, s=y, bl=np.zeros_like(y),
             x_fit=x, y_fit_g=y, y_fit_sn=y, case=1)

    def test_plot_print_and_show(self, sample_data):
        x, y = sample_data
        plot(x, y, show_plot=False, print_plot=False, path="/tmp/test.txt")

    def test_plot_creates_file_on_print(self, sample_data, tmp_path):
        x, y = sample_data
        plot(x, y, print_plot=True, path="/tmp/test.txt",
             output_dir=str(tmp_path))
        expected = tmp_path / "images" / "test.png"
        assert expected.exists()


class TestR2Plots:
    def test_r2_plots_basic(self):
        x = np.geomspace(0.0001, 0.5, 100)
        r2 = np.random.default_rng(0).random(100)
        sm_d0 = np.sort(np.random.default_rng(1).random(100))
        sm_d1 = np.gradient(sm_d0)
        sm_d2 = np.gradient(sm_d1)
        min_d1 = np.array([10, 50])
        max_d1 = np.array([30, 70])
        ends = np.zeros(100, dtype=bool)
        ends[:10] = True
        sec_p = np.array([40, 41, 42, 43], dtype=int)
        tol1_1 = np.zeros(100, dtype=bool)
        tol1_1[:20] = True
        r2_plots(x, r2, sm_d0, sm_d1, sm_d2, min_d1, max_d1,
                 ends, sec_p, tol1_1, 5e-4, 2e-6, 0.01, 0.5, case=1)

    def test_r2_prints(self, tmp_path):
        x = np.geomspace(0.0001, 0.5, 100)
        r2 = np.random.default_rng(0).random(100)
        tol1_1 = np.zeros(100, dtype=bool)
        r2_plots(x, r2, r2, np.gradient(r2), np.gradient(np.gradient(r2)),
                 np.array([], dtype=int), np.array([], dtype=int),
                 np.zeros(100, dtype=bool),
                 np.array([], dtype=int), tol1_1, 5e-4, 2e-6, 0.01, 0.5,
                 print_plot=True, path="/tmp/test.txt",
                 output_dir=str(tmp_path))
        expected = tmp_path / "r2_plots" / "test_r2.png"
        assert expected.exists()

    def test_r2_plots_with_changepoint_overlay(self, tmp_path):
        x = np.geomspace(0.0001, 0.5, 100)
        r2 = np.linspace(1.0, 0.0, 100)
        cp_segments = [
            {'start': 0, 'end': 40, 'flat': True},
            {'start': 40, 'end': 70, 'flat': False},
            {'start': 70, 'end': 100, 'flat': True},
        ]
        r2_plots(x, r2, r2, np.gradient(r2), np.gradient(np.gradient(r2)),
                 np.array([10]), np.array([30]),
                 np.zeros(100, dtype=bool), np.array([40, 41], dtype=int),
                 np.zeros(100, dtype=bool), 5e-4, 2e-6, 0.01, 0.95,
                 cp_segments=cp_segments, cp_chosen=0, cp_fcut=0.01,
                 print_plot=True, path="/tmp/test_cp.txt",
                 output_dir=str(tmp_path))
        expected = tmp_path / "r2_plots" / "test_cp_r2.png"
        assert expected.exists()
