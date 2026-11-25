import numpy as np
import pytest

from weaselytics.utils import (
    _drop_enclosing,
    _durbin_watson,
    _rolling_std,
    end_window,
    l1_trend_filter,
    merge_intervals,
    noise_level,
    peaks_params,
    r2_dw,
    rm_ends_outliers,
)


class TestEndWindow:
    def test_default_parameters(self):
        data = np.zeros(1000)
        size = end_window(data)
        assert size == 10

    def test_clamps_to_minimum(self):
        data = np.zeros(50)
        size = end_window(data, window_min=3, window_max=20)
        assert size == 3

    def test_clamps_to_maximum(self):
        data = np.zeros(5000)
        size = end_window(data, window_min=3, window_max=20)
        assert size == 20


class TestRmEndsOutliers:
    def test_no_outliers(self):
        data = np.ones(100)
        result = rm_ends_outliers(data)
        np.testing.assert_allclose(result, data)

    def test_fixes_first_outlier(self):
        data = np.ones(100)
        data[0] = 100.0
        result = rm_ends_outliers(data)
        assert result[0] != 100.0

    def test_fixes_last_outlier(self):
        data = np.ones(100)
        data[-1] = 100.0
        result = rm_ends_outliers(data)
        assert result[-1] != 100.0


class TestDurbinWatson:
    def test_no_autocorrelation(self):
        rng = np.random.default_rng(42)
        resids = rng.normal(size=1000)
        dw = _durbin_watson(resids)
        assert 1.8 < dw < 2.2

    def test_positive_autocorrelation(self):
        resids = np.arange(100, dtype=float)
        dw = _durbin_watson(resids)
        assert dw < 0.5

    def test_negative_autocorrelation(self):
        resids = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
        dw = _durbin_watson(resids)
        assert dw >= 3.5


class TestR2Dw:
    def test_perfect_correlation(self):
        resids = np.arange(100, dtype=float)
        r2 = r2_dw(resids)
        assert r2 > 0.9

    def test_no_correlation(self):
        rng = np.random.default_rng(42)
        resids = rng.normal(size=1000)
        r2 = r2_dw(resids)
        assert r2 < 0.1

    def test_strong_positive_correlation(self):
        resids = np.array([1, 2, 3, 4], dtype=float)
        r2 = r2_dw(resids)
        assert r2 > 0.9


class TestMergeIntervals:
    def test_non_overlapping(self):
        intervals = np.array([[0, 2], [5, 7]])
        result = merge_intervals(intervals)
        np.testing.assert_array_equal(result, intervals)

    def test_overlapping(self):
        intervals = np.array([[0, 3], [2, 5], [4, 7]])
        result = merge_intervals(intervals)
        np.testing.assert_array_equal(result, [[0, 7]])

    def test_adjacent(self):
        intervals = np.array([[0, 2], [2, 5]])
        result = merge_intervals(intervals)
        np.testing.assert_array_equal(result, [[0, 5]])

    def test_single_interval(self):
        intervals = np.array([[1, 4]])
        result = merge_intervals(intervals)
        np.testing.assert_array_equal(result, [[1, 4]])

    def test_input_is_left_unchanged(self):
        for intervals in ([[0, 5], [3, 9], [20, 25]],
                          np.array([[0, 5], [3, 9], [20, 25]])):
            before = np.array(intervals).copy()
            result = merge_intervals(intervals)
            np.testing.assert_array_equal(result, [[0, 9], [20, 25]])
            np.testing.assert_array_equal(intervals, before)

    def test_empty_input_has_two_columns(self):
        assert merge_intervals([]).shape == (0, 2)
        assert merge_intervals(np.empty((0, 2), dtype=int)).shape == (0, 2)


class TestRollingStd:
    def test_constant_data(self):
        x = np.ones(100)
        rstd = _rolling_std(x, window=5)
        np.testing.assert_allclose(rstd, 0.0, atol=1e-10)

    def test_output_length(self):
        x = np.random.default_rng(0).normal(size=50)
        rstd = _rolling_std(x, window=3)
        assert len(rstd) == len(x)


class TestNoiseLevel:
    def test_white_noise(self):
        s = 0.3 * np.random.default_rng(0).normal(size=20000)
        # The median of the sample standard deviation of 20 points is
        # slightly below the standard deviation of the noise.
        assert np.isclose(noise_level(s), 0.98 * 0.3, rtol=0.03)

    def test_crowded_signal(self):
        # Peaks over about half of the signal.
        t = np.arange(3000.)
        s = 0.1 * np.random.default_rng(1).normal(size=len(t))
        peaks = sum(50. * np.exp(-0.5 * ((t - c) / 9.) ** 2)
                    for c in range(1000, 3000, 160))
        assert np.isclose(noise_level(s + peaks), noise_level(s), rtol=0.1)

    def test_a_drifting_baseline_is_not_noise(self):
        t = np.arange(3000.)
        s = 0.1 * np.random.default_rng(1).normal(size=len(t))
        assert np.isclose(noise_level(s + 1e-6 * (t - 900.)**2),
                          noise_level(s), rtol=0.05)

    def test_slow_noise_is_counted(self):
        white = np.random.default_rng(4).normal(size=20000)
        slow = np.convolve(white, np.ones(4) / 2., mode='same')
        # The part slower than the windows is missing.
        assert 0.7 < noise_level(slow) / slow.std() < 0.9

    def test_one_step_jumps_are_noise(self):
        # A detector whose step is larger than its noise.
        rng = np.random.default_rng(5)
        s = np.where(rng.random(4000) < 0.1, 0.144, 0.)
        assert 0.02 < noise_level(s) < 0.06

    def test_scales_with_the_signal(self):
        s = np.random.default_rng(2).normal(size=3000)
        assert np.isclose(noise_level(250. * s), 250. * noise_level(s))

    def test_noiseless_peaks_have_none(self):
        t = np.arange(1500.)
        s = 20. * np.exp(-0.5 * ((t - 700.) / 8.) ** 2)
        assert noise_level(s) < 1e-12

    @pytest.mark.parametrize("level", [4., 0.1, 12.3, -1e6])
    def test_constant_signal_has_none(self, level):
        assert noise_level(np.full(300, level)) == 0.

    def test_small_peaks_are_found_beside_a_giant_one(self):
        # The giant peak hides the small ones on the first pass.
        t = np.arange(3000.)
        s = 0.1 * np.random.default_rng(6).normal(size=len(t))
        peaks = 1e4 * np.exp(-0.5 * ((t - 200.) / 9.) ** 2)
        peaks += sum(5. * np.exp(-0.5 * ((t - c) / 9.) ** 2)
                     for c in range(500, 3000, 120))
        assert noise_level(s + peaks) < 1.08 * noise_level(s)
        assert noise_level(s + peaks, max_iter=1) > 1.1 * noise_level(s)

    def test_straight_line_has_none(self):
        assert noise_level(0.01 * np.arange(1500.)) < 1e-12

    def test_no_window_free_of_peaks_raises(self):
        # One window only, and a peak in it.
        t = np.arange(300.)
        s = 5. * np.exp(-0.5 * ((t - 150.) / 3.) ** 2)
        with pytest.raises(ValueError, match="free of peaks"):
            noise_level(s, window=300)


class TestL1TrendFilter:
    def test_a_straight_line_is_left_alone(self):
        s = np.linspace(-3., 8., 400)
        np.testing.assert_allclose(l1_trend_filter(s, 0.5, 0.1), s,
                                   atol=1e-6)

    def test_removes_noise(self):
        s = np.random.default_rng(3).normal(size=4000)
        z = l1_trend_filter(s, 6., 1.)
        assert z.std() < 0.3

    def test_keeps_a_tall_narrow_peak(self):
        t = np.arange(400.)
        s = 100. * np.exp(-0.5 * ((t - 200.) / 1.3) ** 2)
        s += np.random.default_rng(4).normal(size=len(t))
        z = l1_trend_filter(s, 6., 1.)
        assert int(z.argmax()) == int(s.argmax())
        assert z.max() > 0.97 * s.max()

    def test_keeps_the_place_of_a_peak_beside_a_deep_dip(self):
        # A Gaussian smoothing of 3 samples moves the top.
        t = np.arange(800.)
        s = 2.5 * np.exp(-0.5 * ((t - 300.) / 1.5) ** 2)
        s -= 6.2 * np.exp(-0.5 * ((t - 304.) / 1.5) ** 2)
        s += 0.02 * np.random.default_rng(2).normal(size=len(t))
        z = l1_trend_filter(s, 0.12, 0.02)
        assert abs(int(z.argmax()) - int(s.argmax())) <= 1
        assert z.max() > 0.95 * s.max()

    def test_no_sample_moves_by_more_than_four_lam(self):
        s = 50. * np.random.default_rng(8).normal(size=500)
        assert np.abs(s - l1_trend_filter(s, 6., 1.)).max() < 4.01 * 6.

    def test_reaches_the_minimum(self):
        rng = np.random.default_rng(5)
        s = np.cumsum(rng.normal(size=300)) * 0.1 + rng.normal(size=300)

        def cost(z):
            return (0.5 * np.sum((s - z) ** 2)
                    + 6. * np.sum(np.abs(np.diff(z, 2))))

        z = l1_trend_filter(s, 6., 1., tol=1e-8, max_iter=5000)
        best = cost(z)
        for _ in range(50):
            step = 1e-3 * np.cumsum(np.cumsum(rng.normal(size=300))) / 300.
            assert cost(z + step) >= best - 1e-4 * best

    def test_scale(self):
        s = np.random.default_rng(6).normal(size=500)
        np.testing.assert_allclose(l1_trend_filter(1e3 * s, 6e3, 1e3),
                                   1e3 * l1_trend_filter(s, 6., 1.),
                                   rtol=1e-6, atol=1e-6)

    def test_input_is_left_unchanged_and_degenerate_cases_copy(self):
        s = np.random.default_rng(7).normal(size=50)
        kept = s.copy()
        for out in (l1_trend_filter(s, 6., 1.), l1_trend_filter(s, 0., 1.),
                    l1_trend_filter(s, 6., 0.), l1_trend_filter(s[:2], 6., 1.)):
            assert out is not s
        np.testing.assert_array_equal(s, kept)
        np.testing.assert_array_equal(l1_trend_filter(s, 0., 1.), s)


class TestDropEnclosing:
    @staticmethod
    def _two(climb=0.):
        t = np.arange(1000.)
        s = 10. * np.exp(-0.5 * ((t - 300.) / 6.) ** 2)
        s += 10.5 * np.exp(-0.5 * ((t - 328.) / 26.) ** 2)
        return s + climb / (1. + np.exp(-(t - 470.) / 6.))

    @pytest.mark.parametrize("climb", [0., 30., 300.])
    def test_a_climbing_baseline_keeps_the_taller_peak(self, climb):
        # The wide shoulder at 328 does not hide the taller peak at 301.
        peaks, _ = peaks_params(self._two(climb), rel_prom_p=0.,
                                drop_enclosing=True, negative=False)
        assert 301 in peaks

    def test_a_wide_feature_enclosing_a_taller_peak_is_dropped(self):
        peaks, widths = np.array([400, 500]), np.array([8., 260.])
        for offset in (0., -50.):
            kept, _ = _drop_enclosing(peaks, widths,
                                      np.array([9., 4.]) + offset)
            assert kept.tolist() == [400]
        # The shoulder of a taller peak is kept.
        kept, _ = _drop_enclosing(peaks, widths, np.array([3., 4.]))
        assert kept.tolist() == [400, 500]


class TestPeaksParamsMinProm:
    @staticmethod
    def _two():
        t = np.arange(600.)
        s = 10. * np.exp(-0.5 * ((t - 200.) / 8.) ** 2)
        return s + 0.5 * np.exp(-0.5 * ((t - 420.) / 8.) ** 2)

    def test_gates_the_positive_peaks(self):
        s = self._two()
        for bar, n in ((0.1, 2), (1., 1), (20., 0)):
            peaks, _ = peaks_params(s, rel_prom_p=0., min_prom=bar)
            assert np.sum(s[peaks] > 0.1) == n

    def test_gates_the_negative_peaks(self):
        s = -self._two()
        kw = dict(rel_prom_p=0., rel_prom_n=0.)
        for bar, n in ((0.1, 2), (1., 1), (20., 0)):
            peaks, _ = peaks_params(s, min_prom=bar, **kw)
            assert np.sum(s[peaks] < -0.1) == n

    def test_negative_peaks_can_be_left_out(self):
        s = self._two() - np.roll(self._two(), 15)
        both, _ = peaks_params(s, rel_prom_p=0., min_prom=0.1)
        positive, widths = peaks_params(s, rel_prom_p=0., min_prom=0.1,
                                        negative=False)
        assert np.any(s[both] < -0.1)
        assert len(positive) == len(widths)
        assert np.all(s[positive] > 0.)

    def test_zero_leaves_the_relative_gates(self):
        s = self._two()
        a, b = peaks_params(s), peaks_params(s, min_prom=0.)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
