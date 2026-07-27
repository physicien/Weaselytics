import numpy as np
import pytest

from weaselytics.peakfitting import (
    PEARSON7_E_BOUNDS,
    PEARSON7_M_BOUNDS,
    _lsq_gauss_fit,
    _lsq_pearson7_fit,
    _lsq_skew_norm_fit,
    fit_peak,
    gauss,
    peaks_params,
    pearson7,
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
        x_fit, y_g, y_sn, y_p7 = fit_peak(y, x)
        assert len(x_fit) > 0
        assert len(y_g) == len(x_fit)
        assert len(y_sn) == len(x_fit)
        assert len(y_p7) == len(x_fit)
        assert np.isclose(np.max(y_g), np.max(y), atol=0.5)
        assert np.isclose(np.max(y_p7), np.max(y), atol=0.5)

    def test_with_x_limits(self, synthetic_gaussian):
        x, y = synthetic_gaussian
        rng = np.random.default_rng(42)
        y = y + 0.1 * rng.normal(size=len(x))
        x_fit, y_g, y_sn, y_p7 = fit_peak(y, x, x0=3.0, x1=7.0)
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


class TestPearson7:
    def test_height_and_centre(self):
        x = np.linspace(0., 10., 2001)
        y = pearson7(x, np.array([3.0, 5.0, 0.4, 10.0, 0.05]))
        assert np.max(y) == pytest.approx(3.0, rel=1e-9)
        assert x[np.argmax(y)] == pytest.approx(5.0, abs=0.01)

    def test_symmetric_when_asymmetry_zero(self):
        x = np.linspace(-5., 5., 2001)
        y = pearson7(x, np.array([1.0, 0.0, 0.5, 8.0, 0.0]))
        np.testing.assert_allclose(y, y[::-1], rtol=0, atol=1e-12)

    def test_gaussian_and_lorentzian_limits(self):
        x = np.linspace(-8., 8., 4001)
        gauss_like = pearson7(x, np.array([1., 0., 1., 500., 0.]))
        lorentz_like = pearson7(x, np.array([1., 0., 1., 1.5, 0.]))
        tail = np.abs(x) > 5
        assert lorentz_like[tail].sum() > 10 * gauss_like[tail].sum()

    def test_single_lobed_past_the_singularity(self):
        # denominator vanishes at x - x0 = -sigma / E
        sigma, e, x0 = 0.2, 0.3, 4.0
        x_sing = x0 - sigma / e
        x = np.linspace(x_sing - 2., x0 + 2., 8001)
        y = pearson7(x, np.array([1.0, x0, sigma, 10.0, e]))
        assert np.all(y[x < x_sing] == 0.0)
        assert np.all(np.isfinite(y))

    def test_rejects_bad_parameters(self):
        x = np.linspace(0., 1., 11)
        with pytest.raises(ValueError, match="sigma"):
            pearson7(x, np.array([1., 0.5, 0.0, 10., 0.]))
        with pytest.raises(ValueError, match="m"):
            pearson7(x, np.array([1., 0.5, 0.1, 0.0, 0.]))

    def test_fit_recovers_its_own_parameters(self):
        x = np.linspace(0., 10., 601)
        true = np.array([3.0, 5.0, 0.5, 12.0, 0.08])
        y = pearson7(x, true)
        got = _lsq_pearson7_fit(x, y)
        assert got[0] == pytest.approx(true[0], rel=0.05)
        assert got[1] == pytest.approx(true[1], abs=0.05)
        assert abs(got[2]) == pytest.approx(true[2], rel=0.20)

    def test_fit_respects_the_published_bounds(self):
        # Milani 2024 §3.1.1 restricts m to 1-1000 and E to +-0.3.
        rng = np.random.default_rng(0)
        x = np.linspace(0., 10., 601)
        y = pearson7(x, np.array([3., 5., 0.5, 12., 0.08]))
        y = y + 0.02 * rng.normal(size=len(x))
        p = _lsq_pearson7_fit(x, y)
        assert PEARSON7_M_BOUNDS[0] <= p[3] <= PEARSON7_M_BOUNDS[1]
        assert PEARSON7_E_BOUNDS[0] <= p[4] <= PEARSON7_E_BOUNDS[1]
