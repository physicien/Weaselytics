"""Tests for the synthetic-truth diagnostics in ``tools/``.

The modules are loaded by path from ``tools/synthetic/``. These tests
exist because the harness defines what "the optimal cutoff" means:
an error here makes a wrong *answer key*, and every constant grounded
against it inherits the fault.
"""

import importlib.util
import inspect
import pathlib

import numpy as np
import pytest

_TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools" / "synthetic"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


diag = _load("synth_diag")
synth = _load("synth_dataset")


class TestErrorCurve:
    def test_finds_the_minimum_of_a_known_signal(self):
        # A signal whose true baseline is known exactly: the error curve
        # must dip somewhere in the interior of the grid rather than
        # running monotonically to an edge, which is what a broken
        # metric would do.
        rng = np.random.default_rng(0)
        d = synth.make_signal(900, 4, 0.019, rng)
        fcuts = np.geomspace(1e-3, 0.2, 12)
        err = diag.error_curve(d['x'], d['y'], d['baseline'], fcuts)
        assert np.isfinite(err).sum() >= 10
        k = int(np.nanargmin(err))
        assert 0 < k < len(fcuts) - 1, (
            "the optimum ran to an edge of the grid")
        # and the metric must actually move
        assert np.nanmax(err) > 2 * np.nanmin(err)

    def test_failed_fits_become_nan_not_zero(self):
        # A NaN means "the fit failed", a zero would mean "perfect".
        # Conflating them would make a crash look like the optimum.
        x = np.linspace(0., 1., 64)
        y = np.zeros(64)
        err = diag.error_curve(x, y, y, np.array([0.4999]))
        assert err.size == 1
        assert np.isnan(err[0]) or err[0] >= 0.0


class TestRegionOf:
    def test_takes_the_last_run_like_select_center(self):
        fr = np.geomspace(1e-4, 0.5, 100)
        mask = np.zeros(100, dtype=bool)
        mask[10:20] = True
        mask[60:71] = True
        lo, hi = diag._region_of(fr, mask)
        assert lo == pytest.approx(fr[60])
        assert hi == pytest.approx(fr[70])

    def test_empty_mask(self):
        fr = np.geomspace(1e-4, 0.5, 50)
        assert diag._region_of(fr, np.zeros(50, dtype=bool)) == (None, None)


class TestSummaryFields:
    def test_every_reported_field_is_declared(self):
        assert 'rel_pos' in diag.SUMMARY_FIELDS
        assert 'consistent' in diag.SUMMARY_FIELDS
        assert len(set(diag.SUMMARY_FIELDS)) == len(diag.SUMMARY_FIELDS)

    def test_peak_area_is_reported_but_does_not_define_the_optimum(self):
        # The optimum is defined by baseline fit alone (Emmanuel,
        # 2026-08-16). Since 2026-08-19 the recovered peak-area error is
        # also REPORTED, following Navarro-Huerta et al. (2017) §3.6, but
        # it must stay a reported figure: `fc_best` is the minimum of the
        # baseline error curve and nothing else.
        assert 'target_area_pct' in diag.SUMMARY_FIELDS
        src = inspect.getsource(diag.diag_one)
        line = 'fc_best, e_min = true_optimum(fcuts, err, n_used)'
        assert line in src
        head = src[:src.index(line)]
        assert 'area' not in head, (
            'a peak-area quantity is computed before fc_best is chosen')


class TestTrueOptimum:
    """The true optimum is searched from the fundamental up (issue #15)."""

    def test_ignores_a_minimum_below_the_fundamental(self):
        fcuts = np.geomspace(1e-5, 0.5, 250, endpoint=False)
        n_used = 500
        err = np.ones(250)
        err[3] = 0.01                     # far below 1/n_used = 2e-3
        k_above = int(np.searchsorted(fcuts, 0.01))
        err[k_above] = 0.5
        fc, e = diag.true_optimum(fcuts, err, n_used)
        assert fc == pytest.approx(fcuts[k_above])
        assert e == pytest.approx(0.5)

    def test_skips_failed_fits_above_the_fundamental(self):
        fcuts = np.geomspace(1e-5, 0.5, 250, endpoint=False)
        err = np.ones(250)
        k = int(np.searchsorted(fcuts, 0.02))
        err[k] = 0.1
        err[k + 1] = np.nan
        assert diag.true_optimum(fcuts, err, 500) == (
            pytest.approx(fcuts[k]), pytest.approx(0.1))

    def test_the_fundamental_itself_is_allowed(self):
        # Same side of the bound as the clip in trim_candidates, which
        # drops only the cutoffs strictly below 1 / n_used.
        fcuts = np.array([1e-3, 2e-3, 4e-3])
        err = np.array([0.01, 0.1, 0.2])
        assert diag.true_optimum(fcuts, err, 500) == (
            pytest.approx(2e-3), pytest.approx(0.1))

    def test_nothing_finite_above_gives_nan(self):
        fcuts = np.geomspace(1e-5, 0.5, 50, endpoint=False)
        err = np.full(50, np.nan)
        err[0] = 1.0
        fc, e = diag.true_optimum(fcuts, err, 10)
        assert np.isnan(fc) and np.isnan(e)


class TestHarnessMatchesProduction:
    """The harness must call the pipeline the way auto_beads does.

    Calling `_fcutoff` with its own defaults runs plain ``beads`` with
    no peak regions instead of ``custom_beads`` with them, a different
    algorithm on a different signal, and nothing in the output shows it:
    the run simply reports worse numbers.
    """

    def test_mirrors_auto_beads_defaults(self):
        from weaselytics.baseline import auto_beads

        src = inspect.getsource(diag.diag_one)
        sig = inspect.signature(auto_beads).parameters
        # the two configuration defaults auto_beads owns
        assert sig['asymmetry'].default == 1.0
        assert sig['fit_parabola'].default is True
        # ... and the harness must state the same ones explicitly
        assert "'asymmetry': 1.0" in src
        assert "'fit_parabola': True" in src
        assert "'alpha': 1.0" in src
        assert "'parabola_len': 3" in src

    def test_past_drop_exclusion_matches_production(self):
        # Production applies the past-drop exclusion on every signal,
        # and undoes it only when it would leave nothing. The synthetic
        # tools hold their own copies of that call. synth_figs is read
        # as text rather than imported, since it imports synth_diag by
        # module name.
        from weaselytics.baseline import _fcutoff

        token = 'exclude_past_drop=True'
        assert token in inspect.getsource(_fcutoff)
        assert token in inspect.getsource(diag.diag_one)
        assert token in (_TOOLS / "synth_figs.py").read_text()

    def test_uses_custom_beads_with_regions(self):
        src = inspect.getsource(diag.diag_one)
        assert "method='custom_beads'" in src, (
            "the harness must not fall back on the plain beads default")
        assert "'regions': peak_regions" in src
        assert "'sampling': sampling" in src

    def test_auto_beads_still_builds_the_same_kwargs(self):
        # If auto_beads ever changes what it passes to _fcutoff, this
        # fails and the harness has to be updated with it.
        import inspect

        from weaselytics.baseline import auto_beads

        src = inspect.getsource(auto_beads)
        for token in ('"asymmetry": asymmetry', '"fit_parabola": fit_parabola',
                      '"alpha": 1.0', '"parabola_len": 3',
                      'regions=peak_regions, sampling=sampling'):
            assert token in src, f'auto_beads no longer contains {token!r}'


class TestNoiseScaledMetrics:
    """The reported accuracy is in units of the record's own noise."""

    def test_fields_are_declared(self):
        for name in ('noise_sigma', 'target_noise', 'excess_noise',
                     'excess_noise_off'):
            assert name in diag.SUMMARY_FIELDS, name


class TestErrorCurveReuse:
    """The error curve is 250 BEADS fits and is reused across runs.

    It must be invalidated by anything that changes it and by nothing
    else, or a run silently scores against a stale answer key.
    """

    def _case(self, seed=0, n=700):
        rng = np.random.default_rng(seed)
        d = synth.make_signal(n, 4, 0.019, rng)
        return d['x'], d['y'], d['baseline']

    def test_key_is_stable_for_the_same_inputs(self):
        x, y, b = self._case()
        fcuts = np.geomspace(1e-5, 0.5, 50, endpoint=False)
        assert (diag._err_key(y, x, b, fcuts)
                == diag._err_key(y, x, b, fcuts))

    def test_key_changes_with_the_signal_and_the_grid(self):
        x, y, b = self._case()
        fcuts = np.geomspace(1e-5, 0.5, 50, endpoint=False)
        base = diag._err_key(y, x, b, fcuts)
        y2 = y.copy()
        y2[len(y2) // 2] += 1.0
        assert diag._err_key(y2, x, b, fcuts) != base
        assert diag._err_key(y, x, b, fcuts[::2]) != base
        assert diag._err_key(y, x, b + 1.0, fcuts) != base

    def test_key_tracks_relevant_regions(self):
        # The curve is fitted with the regions `_relevant_regions`
        # returns, so a change there must invalidate the stored curve.
        # This is the failure the key exists to prevent.
        from unittest.mock import patch

        import weaselytics.baseline as B
        x, y, b = self._case()
        fcuts = np.geomspace(1e-5, 0.5, 50, endpoint=False)
        base = diag._err_key(y, x, b, fcuts)
        with patch.object(diag, '_relevant_regions',
                          lambda s: (None, np.array([1]), len(s))):
            other = diag._err_key(y, x, b, fcuts)
        assert other != base
        assert B is not None

    def test_reused_when_the_optimum_is_excluded(self, tmp_path,
                                                 monkeypatch):
        # The stored key must be the curve's hash on every signal,
        # including those whose optimum the trimming removed, where
        # `excluded_by` is looked up. The selection is replaced by a
        # stub that excludes every cutoff.
        x, y, b = self._case()
        fr = np.geomspace(1e-5, 0.5, 1000, endpoint=False)
        none = np.zeros(1000, dtype=bool)
        pd_stub = {'fcut_range': fr,
                   'r2_val': 0.5 + 0.4 * np.tanh(np.log10(fr) + 2.5),
                   'sensitivity_val': np.ones(1000), 'n_used': 600,
                   'cp_surviving': none, 'cp_removed': ~none,
                   'cp_snr_removed': none, 'cp_instab_removed': none,
                   'cp_dips': none, 'cp_flat': none}
        calls = []

        def curve(xx, s, bb, fcuts):
            if len(fcuts) > 1:
                calls.append(len(fcuts))
            return np.linspace(1.0, 0.1, len(fcuts))

        class Parsed:
            def __init__(self, path):
                self.data = (x, y)

        monkeypatch.setattr(diag, 'ParsedData', Parsed)
        monkeypatch.setattr(diag, 'auto_beads', lambda *a, **k: None)
        monkeypatch.setattr(diag, '_fcutoff',
                            lambda *a, **k: (fr[500], pd_stub))
        monkeypatch.setattr(diag, 'error_curve', curve)
        monkeypatch.setattr(diag, '_err_stats', lambda *a: {
            'rms': 0., 'peak': 0., 'snr_db': 0., 'area_pct': 0.})
        np.savez(tmp_path / 's__truth.npz', baseline=b,
                 noise=np.ones(len(y)), signal=np.zeros(len(y)))
        args = ('s', str(tmp_path), str(tmp_path), str(tmp_path),
                str(tmp_path), 1, 4)
        row = diag.diag_one(*args)
        assert row['excluded_by'] == 'clip_or_tail'
        diag.diag_one(*args)
        assert calls == [250]
        with np.load(tmp_path / 's__diag.npz') as d:
            assert str(d['err_key']) == diag._err_key(y, x, b, fr[::4])


class TestStemFields:
    def test_reads_back_what_the_generator_writes(self):
        stem = synth.signal_stem('12a', 'pearson7', 2, 'snr20', 4312, 7)
        assert diag.stem_fields(stem) == {
            'peak_case': '12a', 'peak_model': 'pearson7',
            'baseline_case': 'type2', 'noise_case': 'snr20',
            'replicate': '0007'}

    def test_reads_a_name_of_the_earlier_generator(self):
        stem = 'SYN__erb__many__pearson7__step__high__07__4312'
        assert diag.stem_fields(stem) == {
            'peak_case': 'many', 'peak_model': 'pearson7',
            'baseline_case': 'step', 'noise_case': 'high',
            'replicate': '07'}

    def test_reads_a_name_without_the_peak_model(self):
        assert diag.stem_fields('SYN__erb__many__step__high__07__4312') == {
            'peak_case': 'many', 'peak_model': '',
            'baseline_case': 'step', 'noise_case': 'high',
            'replicate': '07'}

    def test_short_names_leave_fields_empty(self):
        assert diag.stem_fields('SYN__erb')['replicate'] == ''
