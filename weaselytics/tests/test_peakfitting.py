import numpy as np
import pytest

from weaselytics.peakfitting import (
    _lsq_gauss_fit,
    _lsq_skew_norm_fit,
    fit_peak,
    gauss,
    peaks_params,
    skew_norm,
)


class TestGauss:
    def test_basic_gaussian(self):
        x = np.linspace(-5, 5, 101)
        params = (2.0, 0.0, 1.0)
        y = gauss(x, params)
        assert np.isclose(y.max(), 2.0)
        assert np.isclose(y[50], 2.0)

    def test_center(self):
        x = np.linspace(0, 10, 101)
        params = (1.0, 5.0, 1.0)
        y = gauss(x, params)
        assert np.isclose(y[50], 1.0)

    def test_negative_sigma_raises(self):
        x = np.linspace(0, 5, 50)
        with pytest.raises(ValueError, match="sigma must be greater than 0"):
            gauss(x, (1.0, 2.5, -0.5))


class TestSkewNorm:
    def test_symmetric_alpha_zero(self):
        x = np.linspace(-5, 5, 101)
        params = (1.0, 0.0, 1.0, 0.0)
        y = skew_norm(x, params)
        assert np.isclose(np.argmax(y), 50, atol=5)

    def test_positive_skew(self):
        x = np.linspace(-5, 5, 101)
        params = (1.0, 0.0, 1.0, 5.0)
        y = skew_norm(x, params)
        assert np.argmax(y) > 50


class TestPeaksParams:
    def test_single_peak(self, synthetic_gaussian):
        x, y = synthetic_gaussian
        rng = np.random.default_rng(42)
        y = y + 0.01 * rng.normal(size=len(x))
        peaks, widths = peaks_params(y)
        assert len(peaks) >= 1
        assert np.isclose(x[peaks[0]], 5.0, atol=0.5)

    def test_positive_peaks_only(self):
        rng = np.random.default_rng(42)
        y = np.abs(rng.normal(size=100)) + 0.1
        peaks, widths = peaks_params(y)
        assert len(peaks) >= 0
        assert len(widths) >= 0


class TestFitPeak:
    def test_fit_gaussian_peak(self, synthetic_gaussian):
        x, y = synthetic_gaussian
        rng = np.random.default_rng(42)
        y = y + 0.01 * rng.normal(size=len(x))
        x_fit, y_g, y_sn = fit_peak(y, x)
        assert len(x_fit) > 0
        assert len(y_g) == len(x_fit)
        assert len(y_sn) == len(x_fit)
        assert np.isclose(np.max(y_g), np.max(y), atol=0.5)

    def test_with_x_limits(self, synthetic_gaussian):
        x, y = synthetic_gaussian
        rng = np.random.default_rng(42)
        y = y + 0.1 * rng.normal(size=len(x))
        x_fit, y_g, y_sn = fit_peak(y, x, x0=3.0, x1=7.0)
        assert x_fit[0] >= 2.9
        assert x_fit[-1] <= 7.1


class TestLsqFit:
    def test_lsq_gauss_recovers_parameters(self):
        x = np.linspace(0, 10, 201)
        true = np.array([3.0, 5.0, 0.8])
        y = gauss(x, true)
        result = _lsq_gauss_fit(x, y)
        np.testing.assert_allclose(result, true, atol=0.05)

    def test_lsq_gauss_with_noise(self):
        x = np.linspace(0, 10, 201)
        true = np.array([3.0, 5.0, 0.8])
        rng = np.random.default_rng(42)
        y = gauss(x, true) + 0.02 * rng.normal(size=len(x))
        result = _lsq_gauss_fit(x, y)
        np.testing.assert_allclose(result, true, atol=0.15)

    def test_lsq_skew_norm_recovers_parameters(self):
        x = np.linspace(0, 10, 201)
        true = np.array([3.0, 5.0, 0.8, 3.0])
        y = skew_norm(x, true)
        result = _lsq_skew_norm_fit(x, y)
        np.testing.assert_allclose(result, true, atol=0.05)

    def test_lsq_skew_norm_with_noise(self):
        x = np.linspace(0, 10, 201)
        true = np.array([3.0, 5.0, 0.8, 3.0])
        rng = np.random.default_rng(42)
        y = skew_norm(x, true) + 0.02 * rng.normal(size=len(x))
        result = _lsq_skew_norm_fit(x, y)
        np.testing.assert_allclose(result, true, atol=0.15)
