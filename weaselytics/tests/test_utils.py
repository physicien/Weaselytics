import numpy as np

from weaselytics.utils import (
    _durbin_watson,
    _rolling_std,
    end_window,
    merge_intervals,
    r2_dw,
    rm_ends_outliers,
    smooth_SG,
)


class TestSmoothSG:
    def test_returns_same_length(self):
        x = np.random.default_rng(0).normal(size=100)
        result = smooth_SG(x, 9, 2)
        assert len(result) == len(x)

    def test_smoothes_noisy_data(self):
        x = np.ones(50) + np.random.default_rng(1).normal(0, 0.5, 50)
        result = smooth_SG(x, 11, 3)
        assert np.std(result) < np.std(x)


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


class TestRollingStd:
    def test_constant_data(self):
        x = np.ones(100)
        rstd = _rolling_std(x, window=5)
        np.testing.assert_allclose(rstd, 0.0, atol=1e-10)

    def test_output_length(self):
        x = np.random.default_rng(0).normal(size=50)
        rstd = _rolling_std(x, window=3)
        assert len(rstd) == len(x)
