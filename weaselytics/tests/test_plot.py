import numpy as np

from weaselytics.plot import plot, r2_plots


class TestPlot:
    def test_plot_basic(self, sample_data):
        x, y = sample_data
        plot(x, y)

    def test_plot_with_fits(self, sample_data):
        x, y = sample_data
        plot(x, y, y_sm=y, s=y, bl=np.zeros_like(y),
             x_fit=x, y_fit_g=y, y_fit_sn=y)

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
        rolling_std = np.gradient(np.sort(np.random.default_rng(1).random(100)))
        diff_std_mad = np.gradient(rolling_std)
        r2_plots(x, r2, rolling_std, diff_std_mad, 0.01, 0.5)

    def test_r2_prints(self, tmp_path):
        x = np.geomspace(0.0001, 0.5, 100)
        r2 = np.random.default_rng(0).random(100)
        r2_plots(x, r2, np.gradient(r2), np.gradient(np.gradient(r2)),
                 0.01, 0.5,
                 print_plot=True, path="/tmp/test.txt",
                 output_dir=str(tmp_path))
        expected = tmp_path / "r2_plots" / "test_r2.png"
        assert expected.exists()

    def test_r2_plots_with_changepoint_overlay(self, tmp_path):
        x = np.geomspace(0.0001, 0.5, 100)
        r2 = np.linspace(1.0, 0.0, 100)
        cp_flat = np.zeros(100, dtype=bool)
        cp_flat[20:60] = True
        r2_plots(x, r2, np.gradient(r2), np.gradient(np.gradient(r2)),
                 0.01, 0.95,
                 cp_flat=cp_flat,
                 print_plot=True, path="/tmp/test_cp.txt",
                 output_dir=str(tmp_path))
        expected = tmp_path / "r2_plots" / "test_cp_r2.png"
        assert expected.exists()
