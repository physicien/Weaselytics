import numpy as np
import pytest
from weaselytics.baseline import auto_beads, _beads, _custom_beads
from pybaselines import Baseline


class TestBeads:
    def test_beads_returns_baseline_and_params(self):
        x = np.linspace(0, 10, 101)
        rng = np.random.default_rng(42)
        y = 3.0 * np.exp(-0.5 * ((x - 5.0) / 0.8) ** 2)
        y += 0.1 * (x - 5.0)
        y += 0.01 * rng.normal(size=len(x))
        baseline_fitter = Baseline(x_data=x)
        bl, params = _beads(baseline_fitter, y, freq_cutoff=0.01)
        assert bl is not None
        assert len(bl) == len(y)
        assert "signal" in params

    def test_custom_beads_returns_baseline_and_params(self):
        x = np.linspace(0, 10, 101)
        rng = np.random.default_rng(42)
        y = 3.0 * np.exp(-0.5 * ((x - 5.0) / 0.8) ** 2)
        y += 0.1 * (x - 5.0)
        y += 0.01 * rng.normal(size=len(x))
        baseline_fitter = Baseline(x_data=x)
        bl, params = _custom_beads(baseline_fitter, y, freq_cutoff=0.01)
        assert bl is not None
        assert len(bl) == len(y)
        assert "noise" in params
        assert "signal" in params


class TestAutoBeads:
    def test_with_explicit_freq_cutoff(self):
        x = np.linspace(0, 10, 101)
        rng = np.random.default_rng(42)
        y = 3.0 * np.exp(-0.5 * ((x - 5.0) / 0.8) ** 2)
        y += 0.1 * (x - 5.0)
        y += 0.01 * rng.normal(size=len(x))
        baseline, params, case = auto_beads(
            y, x, freq_cutoff=0.01, method="beads"
        )
        assert len(baseline) == len(y)
        assert case == 0
        assert "signal" in params

    def test_with_custom_beads_method(self):
        x = np.linspace(0, 10, 101)
        rng = np.random.default_rng(42)
        y = 3.0 * np.exp(-0.5 * ((x - 5.0) / 0.8) ** 2)
        y += 0.1 * (x - 5.0)
        y += 0.01 * rng.normal(size=len(x))
        baseline, params, case = auto_beads(
            y, x, freq_cutoff=0.01, method="custom_beads"
        )
        assert len(baseline) == len(y)
        assert case == 0
        assert "noise" in params

    def test_raises_on_invalid_asymmetry(self):
        x = np.linspace(0, 10, 50)
        y = np.ones(50)
        msg = "asymmetry must be greater than 0"
        with pytest.raises(ValueError, match=msg):
            auto_beads(y, x, freq_cutoff=0.01, asymmetry=-1)

    def test_raises_on_invalid_method(self):
        x = np.linspace(0, 10, 50)
        rng = np.random.default_rng(42)
        y = rng.normal(size=50)
        msg = "method 'invalid' is not implemented"
        with pytest.raises(ValueError, match=msg):
            auto_beads(y, x, freq_cutoff=0.01, method="invalid")

    def test_raises_on_invalid_freq_cutoff(self):
        x = np.linspace(0, 10, 50)
        rng = np.random.default_rng(42)
        y = rng.normal(size=50)
        msg = "cutoff frequency must be 0 < freq_cutoff < 0.5"
        with pytest.raises(ValueError, match=msg):
            auto_beads(y, x, freq_cutoff=0.0)
