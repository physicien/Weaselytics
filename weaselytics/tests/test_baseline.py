import numpy as np
import pytest
from pybaselines import Baseline

from weaselytics.baseline import (
    _beads,
    _custom_beads,
    _r2,
    _r2_array,
    _r2_array_cached,
    _relevant_regions,
    auto_beads,
)


class TestRelevantRegions:
    def test_no_relevant_peak_degrades_instead_of_crashing(self):
        # A single broad feature eluting right after the start: its
        # width/x ratio fails the relevance filter, so the relevant
        # set comes out empty and the degraded mode must kick in.
        x = np.arange(1000) / 60.
        rng = np.random.default_rng(7)
        s = 10. * np.exp(-0.5 * ((x - 1.5) / 3.) ** 2)
        s += 0.005 * rng.normal(size=len(x))
        peak_regions, sampling, scut = _relevant_regions(s, x)
        assert peak_regions is None
        assert np.array_equal(sampling, np.array([1]))
        assert scut == len(s)

    def test_spike_on_hump_width_is_not_contaminated(self):
        # A narrow spike riding a broad baseline hump: without the
        # coarse detrend, the half-prominence width of the spike is
        # measured through the hump (hundreds of points), the
        # relevance filter rejects it and the signal degrades. With
        # the detrend the spike keeps its own width and survives.
        x = np.arange(1000) / 60.
        rng = np.random.default_rng(7)
        s = 5. * np.exp(-0.5 * ((x - 8.) / 4.) ** 2)
        s += 3. * np.exp(-0.5 * ((x - 7.) / 0.05) ** 2)
        s += 0.005 * rng.normal(size=len(x))
        peak_regions, sampling, scut = _relevant_regions(s, x)
        assert scut < len(s)

    def test_peaked_signal_still_returns_regions(self):
        x = np.arange(1000) / 60.
        rng = np.random.default_rng(8)
        s = 10. * np.exp(-0.5 * ((x - 6.) / 0.15) ** 2)
        s += 8. * np.exp(-0.5 * ((x - 10.) / 0.4) ** 2)
        s += 0.01 * rng.normal(size=len(x))
        peak_regions, sampling, scut = _relevant_regions(s, x)
        assert peak_regions is not None
        assert scut <= len(s)


class TestBeads:
    def test_beads_returns_baseline_and_params(self):
        x = np.linspace(0, 10, 101)
        rng = np.random.default_rng(100)
        y = 3.0 * np.exp(-0.5 * ((x - 5.0) / 0.8) ** 2)
        y += 0.1 * (x - 5.0)
        y += 0.01 * rng.normal(size=len(x))
        baseline_fitter = Baseline(x_data=x)
        bl, params = _beads(baseline_fitter, y, freq_cutoff=0.01)
        assert bl is not None
        assert len(bl) == len(y)
        assert "signal" in params

    def test_r2_returns_float(self):
        x = np.linspace(0, 10, 101)
        rng = np.random.default_rng(101)
        y = 3.0 * np.exp(-0.5 * ((x - 5.0) / 0.8) ** 2)
        y += 0.1 * (x - 5.0)
        y += 0.01 * rng.normal(size=len(x))
        baseline_fitter = Baseline(x_data=x)
        result = _r2(_beads, baseline_fitter, y, 0.01)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_r2_different_cutoff_gives_different_result(self):
        x = np.linspace(0, 10, 101)
        rng = np.random.default_rng(102)
        y = 3.0 * np.exp(-0.5 * ((x - 5.0) / 0.8) ** 2)
        y += 0.1 * (x - 5.0)
        y += 0.01 * rng.normal(size=len(x))
        baseline_fitter = Baseline(x_data=x)
        r2_low = _r2(_beads, baseline_fitter, y, 0.005)
        r2_high = _r2(_beads, baseline_fitter, y, 0.05)
        assert r2_low != r2_high

    def test_r2_array_returns_correct_length(self):
        x = np.linspace(0, 10, 101)
        rng = np.random.default_rng(103)
        y = 3.0 * np.exp(-0.5 * ((x - 5.0) / 0.8) ** 2)
        y += 0.1 * (x - 5.0)
        y += 0.01 * rng.normal(size=len(x))
        baseline_fitter = Baseline(x_data=x)
        param_range = np.geomspace(0.001, 0.1, 10)
        result = _r2_array(_beads, baseline_fitter, y, param_range)
        assert len(result) == len(param_range)
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_custom_beads_returns_baseline_and_params(self):
        x = np.linspace(0, 10, 101)
        rng = np.random.default_rng(104)
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
        rng = np.random.default_rng(105)
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
        rng = np.random.default_rng(106)
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
        rng = np.random.default_rng(107)
        y = rng.normal(size=50)
        msg = "method 'invalid' is not implemented"
        with pytest.raises(ValueError, match=msg):
            auto_beads(y, x, freq_cutoff=0.01, method="invalid")

    def test_raises_on_invalid_freq_cutoff(self):
        x = np.linspace(0, 10, 50)
        rng = np.random.default_rng(108)
        y = rng.normal(size=50)
        msg = "cutoff frequency must be 0 < freq_cutoff < 0.5"
        with pytest.raises(ValueError, match=msg):
            auto_beads(y, x, freq_cutoff=0.0)


class TestR2ArrayCached:
    def _setup(self):
        x = np.linspace(0, 10, 101)
        rng = np.random.default_rng(104)
        y = 3.0 * np.exp(-0.5 * ((x - 5.0) / 0.8) ** 2)
        y += 0.1 * (x - 5.0)
        y += 0.01 * rng.normal(size=len(x))
        baseline_fitter = Baseline(x_data=x)
        param_range = np.geomspace(0.001, 0.1, 10)
        return baseline_fitter, y, param_range

    def test_no_cache_dir_matches_uncached(self):
        baseline_fitter, y, param_range = self._setup()
        expected = _r2_array(_beads, baseline_fitter, y, param_range)
        result = _r2_array_cached(_beads, baseline_fitter, y, param_range)
        assert np.allclose(result, expected)

    def test_cache_roundtrip(self, tmp_path):
        baseline_fitter, y, param_range = self._setup()
        cache_dir = str(tmp_path / "cache")
        cold = _r2_array_cached(
            _beads, baseline_fitter, y, param_range,
            cache_dir=cache_dir, path="./sample.txt")
        cache_files = list((tmp_path / "cache").glob("sample__r2__*.npz"))
        assert len(cache_files) == 1
        warm = _r2_array_cached(
            _beads, baseline_fitter, y, param_range,
            cache_dir=cache_dir, path="./sample.txt")
        assert np.array_equal(cold, warm)
        # No second file was written on the warm call
        assert len(list((tmp_path / "cache").glob("*.npz"))) == 1

    def test_cache_file_is_self_contained(self, tmp_path):
        baseline_fitter, y, param_range = self._setup()
        cache_dir = str(tmp_path / "cache")
        r2_val = _r2_array_cached(
            _beads, baseline_fitter, y, param_range,
            cache_dir=cache_dir, path="./sample.txt")
        cache_file = next((tmp_path / "cache").glob("*.npz"))
        with np.load(cache_file) as data:
            assert np.array_equal(data["fcut_range"], param_range)
            assert np.array_equal(data["r2_val"], r2_val)

    def test_new_kwargs_evict_stale_entry(self, tmp_path):
        baseline_fitter, y, param_range = self._setup()
        cache_dir = str(tmp_path / "cache")
        _r2_array_cached(
            _beads, baseline_fitter, y, param_range,
            cache_dir=cache_dir, path="./sample.txt", asymmetry=1.0)
        old_file = next((tmp_path / "cache").glob("*.npz"))
        _r2_array_cached(
            _beads, baseline_fitter, y, param_range,
            cache_dir=cache_dir, path="./sample.txt", asymmetry=6.0)
        # The stale entry was replaced, not accumulated
        cache_files = list((tmp_path / "cache").glob("*.npz"))
        assert len(cache_files) == 1
        assert cache_files[0] != old_file
        # The new entry is a valid cache hit for the new kwargs
        warm = _r2_array_cached(
            _beads, baseline_fitter, y, param_range,
            cache_dir=cache_dir, path="./sample.txt", asymmetry=6.0)
        assert len(list((tmp_path / "cache").glob("*.npz"))) == 1
        assert len(warm) == len(param_range)

    def test_new_signal_evicts_stale_entry(self, tmp_path):
        baseline_fitter, y, param_range = self._setup()
        cache_dir = str(tmp_path / "cache")
        _r2_array_cached(
            _beads, baseline_fitter, y, param_range,
            cache_dir=cache_dir, path="./sample.txt")
        _r2_array_cached(
            _beads, baseline_fitter, y + 0.1, param_range,
            cache_dir=cache_dir, path="./sample.txt")
        assert len(list((tmp_path / "cache").glob("*.npz"))) == 1

    def test_eviction_only_touches_same_stem(self, tmp_path):
        baseline_fitter, y, param_range = self._setup()
        cache_dir = str(tmp_path / "cache")
        _r2_array_cached(
            _beads, baseline_fitter, y, param_range,
            cache_dir=cache_dir, path="./sample.txt")
        _r2_array_cached(
            _beads, baseline_fitter, y, param_range,
            cache_dir=cache_dir, path="./other.txt")
        # One entry per data file is kept
        assert len(list((tmp_path / "cache").glob("sample__r2__*.npz"))) == 1
        assert len(list((tmp_path / "cache").glob("other__r2__*.npz"))) == 1


class TestR2ArrayParallel:
    def test_parallel_matches_serial(self):
        x = np.linspace(0, 10, 101)
        rng = np.random.default_rng(105)
        y = 3.0 * np.exp(-0.5 * ((x - 5.0) / 0.8) ** 2)
        y += 0.1 * (x - 5.0)
        y += 0.01 * rng.normal(size=len(x))
        baseline_fitter = Baseline(x_data=x)
        param_range = np.geomspace(0.001, 0.1, 10)
        serial = _r2_array(_beads, baseline_fitter, y, param_range)
        parallel = _r2_array(_beads, baseline_fitter, y, param_range,
                             workers=2)
        assert len(parallel) == len(param_range)
        assert np.allclose(parallel, serial)

    def test_more_workers_than_params(self):
        x = np.linspace(0, 10, 101)
        rng = np.random.default_rng(106)
        y = 3.0 * np.exp(-0.5 * ((x - 5.0) / 0.8) ** 2)
        y += 0.01 * rng.normal(size=len(x))
        baseline_fitter = Baseline(x_data=x)
        param_range = np.geomspace(0.001, 0.1, 3)
        serial = _r2_array(_beads, baseline_fitter, y, param_range)
        parallel = _r2_array(_beads, baseline_fitter, y, param_range,
                             workers=8)
        assert np.allclose(parallel, serial)
