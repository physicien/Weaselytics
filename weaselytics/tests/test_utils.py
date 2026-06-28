import numpy as np
import pytest

from weaselytics.utils import (
    _durbin_watson,
    _long_segments,
    _rolling_mad,
    _rolling_std,
    continuous_ranges,
    end_window,
    find_flat,
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


class TestContinuousRanges:
    def test_single_range(self):
        x = np.array([1, 2, 3, 4, 5])
        result = continuous_ranges(x)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], x)

    def test_two_ranges(self):
        x = np.array([1, 2, 3, 10, 11, 12])
        result = continuous_ranges(x)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], [1, 2, 3])
        np.testing.assert_array_equal(result[1], [10, 11, 12])

    def test_empty_input(self):
        x = np.array([])
        result = continuous_ranges(x)
        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) == 0


class TestFindFlat:
    def test_absolute_mode(self):
        x = np.array([0.01, 0.02, 0.5, 0.03, 0.01])
        result = find_flat(x, include_tol=0.1, mode='absolute')
        expected = np.where((np.abs(x) < 0.1))[0]
        np.testing.assert_array_equal(result, expected)

    def test_signed_mode(self):
        x = np.array([-0.05, 0.01, 0.5, -0.03, 0.02])
        result = find_flat(x, include_tol=0.1, mode='signed')
        expected = np.where((x < 0.1) & (x > 0))[0]
        np.testing.assert_array_equal(result, expected)

    def test_exclude_tol(self):
        x = np.array([0.001, 0.01, 0.5, 0.02, 0.001])
        result = find_flat(x, include_tol=0.1, exclude_tol=0.005)
        expected = np.where((x < 0.1) & (x > 0.005))[0]
        np.testing.assert_array_equal(result, expected)

    def test_invalid_mode(self):
        msg = "mode 'invalid' is not supported"
        with pytest.raises(ValueError, match=msg):
            find_flat(np.array([0.1, 0.2]), 0.5, mode='invalid')

    def test_exclude_greater_than_include(self):
        msg = "exclude_tol must be smaller than include_tol"
        with pytest.raises(ValueError, match=msg):
            find_flat(np.array([0.1, 0.2]), 0.1, exclude_tol=0.2)


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


class TestRollingMad:
    def test_constant_data(self):
        x = np.ones(100)
        rmad = _rolling_mad(x, window=5)
        np.testing.assert_allclose(rmad, 0.0, atol=1e-10)

    def test_output_length(self):
        x = np.random.default_rng(0).normal(size=50)
        rmad = _rolling_mad(x, window=3)
        assert len(rmad) == len(x)


class TestLongSegments:
    def test_keeps_long_true_segments(self):
        x = np.array([False, True, True, True, False, True, False])
        result = _long_segments(x, min_len=2)
        expected = np.array([False, True, True, True, False, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_removes_short_segments(self):
        x = np.array([True, False, True, True, False])
        result = _long_segments(x, min_len=3)
        expected = np.array([False, False, False, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_all_true(self):
        x = np.ones(10, dtype=bool)
        result = _long_segments(x, min_len=5)
        np.testing.assert_array_equal(result, x)
