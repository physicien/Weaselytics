from unittest import mock

import numpy as np
import pytest
from pybaselines import Baseline

from weaselytics import baseline as baseline_module
from weaselytics.baseline import (
    _R2_CHANNEL,
    _beads,
    _custom_beads,
    _r2,
    _r2_array,
    _r2_array_cached,
    _r2_cache_key,
    _relevant_regions,
    _snr,
    auto_beads,
)
from weaselytics.utils import r2_dw


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


class TestSnr:
    def test_high_for_analyte_low_for_blank(self):
        # A tall peak on light noise is well above the gate; a flat
        # trace of the same noise is well below it. The ~25 split must
        # sit cleanly between them.
        x = np.linspace(0, 10, 1000)
        rng = np.random.default_rng(3)
        peak = 5. * np.exp(-0.5 * ((x - 5.) / 0.1) ** 2)
        peak += 0.01 * rng.normal(size=1000)
        blank = 0.01 * rng.normal(size=1000)
        assert _snr(peak) >= 25.
        assert _snr(blank) < 25.

    def test_constant_difference_signal_is_infinite(self):
        # A perfectly linear ramp has constant consecutive differences,
        # so the MAD-of-differences noise estimate is zero.
        assert _snr(np.linspace(0., 1., 500)) == np.inf


class TestFcutoffDegenerateCurves:
    """The legacy derivative route must fail legibly, not on an
    IndexError, when its absolute tolerances find nothing to anchor on.

    Both conditions occur in practice: on the 72-signal synthetic
    benchmark 7 signals crashed this way, 6 with an empty
    secondary-plateau set and 1 with no initial plateau.
    """

    _N = 1000

    def _drive(self, r2):
        x = np.linspace(0, 10, 200)
        s = np.exp(-0.5 * ((x - 5.0) / 0.5) ** 2)
        with mock.patch.object(baseline_module, "_r2_array_cached",
                               return_value=np.ascontiguousarray(r2)):
            return baseline_module._fcutoff(s, x, len(s), num=self._N,
                                            method="beads")

    def test_no_initial_plateau_raises_a_described_error(self):
        # A curve that starts descending immediately: no point before
        # the steepest descent sits within tol0 of the level of the
        # first flat run.
        t = np.linspace(0, 1, self._N)
        with pytest.raises(ValueError, match="no initial plateau found"):
            self._drive(0.999 * np.exp(-5.0 * t))

    def test_no_secondary_plateau_raises_a_described_error(self):
        # A rippled step. The ripple puts the d1-flat and d2-flat sets
        # in antiphase -- the second derivative vanishes exactly where
        # the slope peaks -- so their intersection is empty even though
        # both sets are large.
        t = np.linspace(0, 1, self._N)
        r2 = (0.999 - 0.5 / (1 + np.exp(-(t - 0.55) * 25))
              + 0.03 * np.sin(2 * np.pi * 14 * t))
        with pytest.raises(ValueError, match="no secondary plateau found"):
            self._drive(r2)

    def test_a_well_formed_curve_still_selects(self):
        t = np.linspace(0, 1, self._N)
        r2 = 0.999 - 0.5 / (1 + np.exp(-(t - 0.55) * 25))
        fcut, case, plot_data = self._drive(r2)
        assert 0.0 < fcut < 0.5
        assert case in (1, 2)


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

    def test_r2_is_computed_on_the_baseline_corrected_signal(self):
        # `r2` must be the Durbin-Watson statistic of `y - baseline`
        # (the corrected signal `c + e` of Navarro-Huerta Eq. 12), not
        # of the denoised `params["signal"]`. The noise is the white
        # floor the drop is measured against; removing it removes what
        # makes the statistic diagnostic.
        x = np.linspace(0, 10, 101)
        rng = np.random.default_rng(104)
        y = 3.0 * np.exp(-0.5 * ((x - 5.0) / 0.8) ** 2)
        y += 0.1 * (x - 5.0)
        y += 0.01 * rng.normal(size=len(x))
        baseline_fitter = Baseline(x_data=x)
        bl, params = _beads(baseline_fitter, y, freq_cutoff=0.01)
        assert _r2(_beads, baseline_fitter, y, 0.01) == pytest.approx(
            r2_dw(y - bl))
        # and specifically not the denoised component
        assert _r2(_beads, baseline_fitter, y, 0.01) != pytest.approx(
            r2_dw(params["signal"]))

    def test_r2_agrees_between_beads_and_custom_beads(self):
        # Both paths must correlate the same quantity, so that plateau
        # logic tuned on one transfers to the other. This failed before
        # the channel was changed: `_custom_beads` rebuilt
        # `params["signal"]` with a noise term interpolated from the
        # reduced grid, exact outside the regions and absent inside
        # them, leaving raw point-to-point noise on the interiors.
        #
        # The fixture is blank-like on purpose. The failure scaled as
        # (region coverage x noise) / analyte amplitude, so a strong
        # peak hides it entirely; with a negligible analyte the old
        # channel gave 0.020 against 0.924 for `beads`.
        x = np.linspace(0, 40, 1200)
        rng = np.random.default_rng(105)
        y = 0.03 * np.exp(-0.5 * ((x - 30.0) / 2.5) ** 2)
        y += 0.5 * np.exp(-x / 12.0)
        y += 0.01 * rng.normal(size=len(x))
        baseline_fitter = Baseline(x_data=x)
        plain = _r2(_beads, baseline_fitter, y, 0.01)
        custom = _r2(_custom_beads, baseline_fitter, y, 0.01,
                     regions=np.array([[820, 980]]),
                     sampling=np.array([8]))
        assert custom == pytest.approx(plain, abs=0.05)

    def test_r2_cache_key_survives_a_one_ulp_grid_difference(self):
        # `np.geomspace` is not bit-reproducible across numpy versions
        # and platforms: between this machine and the cluster, 48 of
        # 1000 grid values differed in their last bit. Hashing the raw
        # float64 bytes made the cache unshareable, so the sweep was
        # recomputed on every machine.
        signal = np.linspace(0.0, 1.0, 200)
        grid = np.geomspace(1e-5, 0.5, 1000, endpoint=False)
        nudged = grid.copy()
        nudged[::20] = np.nextafter(nudged[::20], np.inf)
        assert not np.array_equal(grid, nudged)
        assert _r2_cache_key(_beads, signal, grid, "freq_cutoff", {}) == \
            _r2_cache_key(_beads, signal, nudged, "freq_cutoff", {})

    def test_r2_cache_key_survives_a_one_ulp_signal_difference(self):
        # Same hazard on the other array, and the one the first attempt
        # missed: the signal arrives as `log10(s - min(s) + eps)`, and
        # `np.log10` is not identical across libm implementations. With
        # the grid quantised but the signal still hashed as float64,
        # the cluster's cache still missed locally on all four signals
        # checked.
        grid = np.geomspace(1e-5, 0.5, 1000, endpoint=False)
        signal = np.log10(np.linspace(0.0, 4.5, 3000) + 1.0)
        nudged = signal.copy()
        nudged[::7] = np.nextafter(nudged[::7], np.inf)
        assert not np.array_equal(signal, nudged)
        assert _r2_cache_key(_beads, signal, grid, "freq_cutoff", {}) == \
            _r2_cache_key(_beads, nudged, grid, "freq_cutoff", {})

    def test_r2_cache_key_separates_a_different_truncation(self):
        # A different `scut` must never collide, however close the
        # retained values are: the lengths are hashed explicitly.
        grid = np.geomspace(1e-5, 0.5, 1000, endpoint=False)
        signal = np.log10(np.linspace(0.0, 4.5, 3000) + 1.0)
        assert _r2_cache_key(_beads, signal, grid, "freq_cutoff", {}) != \
            _r2_cache_key(_beads, signal[:-1], grid, "freq_cutoff", {})

    def test_r2_cache_key_still_separates_real_signals(self):
        grid = np.geomspace(1e-5, 0.5, 1000, endpoint=False)
        rng = np.random.default_rng(11)
        base_signal = np.log10(np.linspace(0.0, 4.5, 3000) + 1.0)
        base = _r2_cache_key(_beads, base_signal, grid, "freq_cutoff", {})
        # a perturbation far below detector noise but far above float32
        other = base_signal + 1e-5 * rng.normal(size=len(base_signal))
        assert _r2_cache_key(_beads, other, grid, "freq_cutoff", {}) != base

    def test_r2_cache_key_still_separates_real_grid_changes(self):
        signal = np.linspace(0.0, 1.0, 200)
        base = _r2_cache_key(
            _beads, signal, np.geomspace(1e-5, 0.5, 1000, endpoint=False),
            "freq_cutoff", {})
        for other in (np.geomspace(1e-5, 0.5, 999, endpoint=False),
                      np.geomspace(2e-5, 0.5, 1000, endpoint=False),
                      np.geomspace(1e-5, 0.4, 1000, endpoint=False)):
            assert _r2_cache_key(_beads, signal, other,
                                 "freq_cutoff", {}) != base

    def test_r2_cache_key_tracks_the_channel(self):
        # The channel is not an input to the fit, so a change to it
        # leaves signal, param_range and kwargs untouched. Without the
        # channel token in the key, stale curves would be served.
        signal = np.linspace(0.0, 1.0, 50)
        param_range = np.geomspace(1e-4, 0.4, 10)
        key = _r2_cache_key(_beads, signal, param_range, "freq_cutoff", {})
        assert _R2_CHANNEL in ("y-baseline",)
        import unittest.mock as mock
        with mock.patch.object(baseline_module, "_R2_CHANNEL", "other"):
            other = _r2_cache_key(_beads, signal, param_range,
                                  "freq_cutoff", {})
        assert key != other

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
