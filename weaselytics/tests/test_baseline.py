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
    def test_peaked_signal_still_returns_regions(self):
        x = np.arange(1000) / 60.
        rng = np.random.default_rng(8)
        s = 10. * np.exp(-0.5 * ((x - 6.) / 0.15) ** 2)
        s += 8. * np.exp(-0.5 * ((x - 10.) / 0.4) ** 2)
        s += 0.01 * rng.normal(size=len(x))
        peak_regions, sampling, scut = _relevant_regions(s, x)
        assert peak_regions is not None
        assert scut <= len(s)


class TestSmoothSigma:
    """The pre-smoothing width is an argument, and it is a sigma.

    Exposing it changed no result. Checked by running the previous
    revision and this one over the 339 LPYE records and comparing all
    three outputs: `peak_regions`, `sampling` and `scut` agree on every
    record. That comparison needs two revisions of the module and so
    cannot live here; what is asserted below is that the default is
    still 3 and that the argument reaches the filter.

    It is not a window length, so a small value does not clean the
    trace, it stops cleaning it.
    """

    @staticmethod
    def _noisy_peak():
        x = np.arange(2000) / 60.
        rng = np.random.default_rng(11)
        s = 5. * np.exp(-0.5 * ((x - 12.) / 0.2) ** 2)
        s += 0.05 * rng.normal(size=len(x))
        return x, s

    def test_default_is_three(self):
        import inspect

        sig = inspect.signature(_relevant_regions)
        assert sig.parameters['smooth_sigma'].default == 3.

    def test_passing_the_default_explicitly_is_a_no_op(self):
        x, s = self._noisy_peak()
        assert (_relevant_regions(s, x)[2]
                == _relevant_regions(s, x, smooth_sigma=3.)[2])

    def test_a_small_sigma_admits_the_noise(self):
        # At a spike-removal scale the detection reads the noise floor
        # as peaks; measured over the 339 LPYE records, the relevance
        # filter returns 2092 against 1294.
        x, s = self._noisy_peak()
        wide = _relevant_regions(s, x, smooth_sigma=3.)
        narrow = _relevant_regions(s, x, smooth_sigma=0.5)
        assert narrow[2] > wide[2]


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


class TestFcutoffSelection:
    """Stage 3 reduces the surviving plateau to a single cutoff.

    The record is deliberately long: the fundamental is `1 / n_used`,
    so a short one puts the sub-fundamental clip ABOVE the plateau of
    the test curve and stage 2 legitimately removes everything, which
    would test the wrong thing.
    """

    _N = 1000

    def _drive(self, r2):
        x = np.linspace(0, 10, 4000)
        s = np.exp(-0.5 * ((x - 5.0) / 0.5) ** 2)
        # _fcutoff requests the Durbin-Watson array and the sensitivity
        # curve too, so the mock returns the (r2, dw, sensitivity) tuple.
        # Neither extra is used by the assertions here; dw is filled with
        # 2.0, the value at which the residual is uncorrelated.
        with mock.patch.object(
                baseline_module, "_r2_array_cached",
                return_value=(np.ascontiguousarray(r2),
                              np.full_like(r2, 2.0, dtype=float),
                              np.zeros_like(r2, dtype=float))):
            return baseline_module._fcutoff(s, x, len(s), num=self._N,
                                            method="beads")

    def test_a_well_formed_curve_selects_the_centre(self):
        t = np.linspace(0, 1, self._N)
        r2 = 0.999 - 0.5 / (1 + np.exp(-(t - 0.55) * 25))
        fcut, plot_data = self._drive(r2)
        assert 0.0 < fcut < 0.5
        # the cutoff is the geometric centre of what stage 2 left
        surviving = plot_data["cp_surviving"]
        grid = plot_data["fcut_range"]
        idx = np.flatnonzero(surviving)
        assert idx.size
        splits = np.where(np.diff(idx) > 1)[0] + 1
        region = np.split(idx, splits)[-1]
        # The centre is taken on the index axis and therefore lands on a
        # grid point the sweep evaluated, so it equals the geometric mean
        # of the region's ends only when their sum is even. Comparing the
        # index avoids asserting a parity accident.
        centre = int(round(0.5 * (region[0] + region[-1])))
        assert fcut == pytest.approx(grid[centre])
        step = grid[1] / grid[0]
        assert fcut / np.sqrt(grid[region[0]] * grid[region[-1]]) == (
            pytest.approx(1.0, rel=step - 1.0))

    def test_no_surviving_plateau_raises_a_described_error(self):
        # A pure descent: nothing is flat, so stage 2 leaves nothing and
        # there is no region to take the centre of. Failing loudly beats
        # substituting a cutoff -- a wrong fcut silently biases every
        # area derived from it.
        t = np.linspace(0, 1, self._N)
        with pytest.raises(ValueError, match="no surviving plateau"):
            self._drive(0.999 - 0.9 * t)


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
        assert _R2_CHANNEL in ("y-baseline-dw",)
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
        baseline, params = auto_beads(
            y, x, freq_cutoff=0.01, method="beads"
        )
        assert len(baseline) == len(y)
        assert "signal" in params

    def test_with_custom_beads_method(self):
        x = np.linspace(0, 10, 101)
        rng = np.random.default_rng(106)
        y = 3.0 * np.exp(-0.5 * ((x - 5.0) / 0.8) ** 2)
        y += 0.1 * (x - 5.0)
        y += 0.01 * rng.normal(size=len(x))
        baseline, params = auto_beads(
            y, x, freq_cutoff=0.01, method="custom_beads"
        )
        assert len(baseline) == len(y)
        assert "noise" in params

    def test_default_method_is_custom_beads(self):
        """The default fits through custom_bc, not the plain path.

        Every other test passes `method` explicitly, so nothing else
        here notices if the default moves. It matters: the two paths
        put the collapse of the r2 curve at different cutoffs, and the
        selection anchors on the last plateau before it.

        Routing is asserted directly rather than through the fitted
        baseline, because the two paths return the identical baseline
        whenever ``_relevant_regions`` finds no region, which is the
        case on small synthetic traces.
        """
        x = np.linspace(0, 10, 201)
        rng = np.random.default_rng(107)
        y = 3.0 * np.exp(-0.5 * ((x - 3.0) / 0.3) ** 2)
        y += 2.0 * np.exp(-0.5 * ((x - 7.0) / 1.2) ** 2)
        y += 0.1 * (x - 5.0)
        y += 0.01 * rng.normal(size=len(x))

        import weaselytics.baseline as bl_mod
        called = []
        real = bl_mod._custom_beads

        def spy(*args, **kwargs):
            called.append(kwargs.get('regions', 'missing'))
            return real(*args, **kwargs)

        bl_mod._custom_beads = spy
        # The dispatch table is rebuilt per call from module globals.
        try:
            baseline, params = auto_beads(y, x, freq_cutoff=0.01)
        finally:
            bl_mod._custom_beads = real

        assert called, "auto_beads did not route through _custom_beads"
        assert len(baseline) == len(y)
        # `noise` is rebuilt only on the custom_bc path.
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


class TestSensitivityCurve:
    def _setup(self):
        x = np.linspace(0, 10, 101)
        rng = np.random.default_rng(107)
        y = 3.0 * np.exp(-0.5 * ((x - 5.0) / 0.8) ** 2)
        y += 0.1 * (x - 5.0)
        y += 0.01 * rng.normal(size=len(x))
        return Baseline(x_data=x), y, np.geomspace(0.001, 0.1, 12)

    def test_shape_first_zero_and_nonnegative(self):
        fitter, y, pr = self._setup()
        r2, stab = _r2_array(_beads, fitter, y, pr, return_sensitivity=True)
        assert stab.shape == r2.shape == pr.shape
        assert stab[0] == 0.0
        assert np.all(stab >= 0.0)
        assert np.all(np.isfinite(stab))

    def test_default_returns_only_r2(self):
        fitter, y, pr = self._setup()
        out = _r2_array(_beads, fitter, y, pr)
        assert isinstance(out, np.ndarray)

    def test_parallel_matches_serial(self):
        # The seam stitching across worker chunks must reproduce the
        # serial step-to-step baseline change exactly.
        fitter, y, pr = self._setup()
        _, s_serial = _r2_array(_beads, fitter, y, pr, return_sensitivity=True)
        _, s_par = _r2_array(_beads, fitter, y, pr, workers=3,
                             return_sensitivity=True)
        assert np.allclose(s_par, s_serial)

    def test_cache_stores_and_restores_sensitivity(self, tmp_path):
        fitter, y, pr = self._setup()
        cd = str(tmp_path / "cache")
        r2c, sc = _r2_array_cached(_beads, fitter, y, pr, cache_dir=cd,
                                   path="./s.txt", return_sensitivity=True)
        r2w, sw = _r2_array_cached(_beads, fitter, y, pr, cache_dir=cd,
                                   path="./s.txt", return_sensitivity=True)
        assert np.array_equal(sc, sw)
        assert np.array_equal(r2c, r2w)
        cache_file = list((tmp_path / "cache").glob("*.npz"))[0]
        with np.load(cache_file) as d:
            assert "sensitivity" in d.files

class TestRelevantRegionsHasNoDetrend:
    """`_relevant_regions` measures widths on the raw smoothed curve.

    An N/4 rolling-median detrend (6a1a380) was removed on 2026-08-17.
    Measured against the true widths of 1520 peaks over 576 synthetic
    chromatograms -- truth taken as the apex and FWHM of the noise-free
    EMG profiles, not the stored Gaussian parameters -- it was worse
    than the raw curve on median error (4.83 vs 4.22), on p90 (84 vs
    45) and on bias (-17.1 vs +4.9), and it biased widths NARROW, which
    silently drops real peaks. Reproducing the decision the relevance
    filter would make with exact widths, it cost 224 errors against 133.
    """

    def test_no_median_detrend(self):
        import inspect

        from weaselytics.baseline import _relevant_regions

        src = '\n'.join(ln for ln in
                        inspect.getsource(_relevant_regions).splitlines()
                        if not ln.lstrip().startswith('#'))
        assert 'median_filter' not in src, (
            'the rolling-median detrend is back; it measures widths '
            'narrower than the truth and drops real peaks')

    def test_structure_carrying_taller_peaks_is_discarded(self):
        import inspect

        from weaselytics.baseline import _relevant_regions

        src = inspect.getsource(_relevant_regions)
        assert 'drop_enclosing=True' in src, (
            'a feature carrying taller peaks must not define a region; '
            'the peaks on it must')
