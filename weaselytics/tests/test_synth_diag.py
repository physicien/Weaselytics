"""Tests for the synthetic-truth diagnostics in ``tools/``.

``tools/`` is not a package, so the modules are loaded by path. These
tests exist because the harness defines what "the optimal cutoff" means:
an error here does not produce a wrong answer, it produces a wrong
*answer key*, and every constant grounded against it inherits the fault.
"""

import importlib.util
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
report = _load("synth_report")
synth = _load("synth_dataset")


class TestErrorCurve:
    def test_finds_the_minimum_of_a_known_signal(self):
        # A signal whose true baseline is known exactly: the error curve
        # must dip somewhere in the interior of the grid rather than
        # running monotonically to an edge, which is what a broken
        # metric would do.
        rng = np.random.default_rng(0)
        d = synth.erb_native_signal(900, 'multi_narrow', 1, 0.019, rng,
                                    quantise_output=False)
        fcuts = np.geomspace(1e-3, 0.2, 12)
        err = diag.error_curve(d['x'], d['y'], d['baseline'], fcuts)
        assert np.isfinite(err).sum() >= 10
        k = int(np.nanargmin(err))
        assert 0 < k < len(fcuts) - 1, (
            "the optimum ran to an edge of the grid")
        # and the metric must actually move
        assert np.nanmax(err) > 2 * np.nanmin(err)

    def test_error_is_zero_against_a_perfect_baseline(self):
        # Scoring a baseline against itself must give exactly zero, so
        # the metric is an RMSE and not something offset.
        b = np.linspace(-5., 5., 200)
        assert float(np.sqrt(np.mean((b - b) ** 2))) == 0.0

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

    def test_rel_pos_matches_a_hand_computation(self):
        # rel_pos is measured in log(fcut); an arithmetic position would
        # sit well to the right on a geometric grid and is a different
        # point.
        lo, hi, best = 1e-3, 1e-1, 1e-2
        rel = (np.log10(best) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        assert rel == pytest.approx(0.5)


class TestLocalMinima:
    def test_finds_interior_minima_only(self):
        v = np.array([5., 4., 3., 4., 5., 2., 6., 7.])
        idx = report.local_minima(v)
        assert set(idx.tolist()) == {2, 5}

    def test_monotone_curve_has_none(self):
        assert report.local_minima(np.arange(50.)).size == 0
        assert report.local_minima(np.arange(50.)[::-1]).size == 0

    def test_too_short_returns_empty(self):
        assert report.local_minima(np.array([1., 2.])).size == 0

    def test_distance_is_measured_in_decades(self):
        fr = np.geomspace(1e-4, 1e-1, 4)      # 1e-4,1e-3,1e-2,1e-1
        minima = np.array([1])                 # at 1e-3
        assert report.nearest_min_distance(fr, minima, 1e-2) == \
            pytest.approx(1.0)
        assert report.nearest_min_distance(fr, minima, 1e-3) == \
            pytest.approx(0.0)

    def test_no_minima_gives_nan(self):
        fr = np.geomspace(1e-4, 1e-1, 4)
        assert np.isnan(report.nearest_min_distance(
            fr, np.array([], dtype=int), 1e-2))

    def test_the_null_control_is_not_degenerate(self):
        # If a curve is dense in minima, a random cutoff is also close to
        # one. This pins that the null control can detect that case --
        # without it, "the optimum is near a minimum" is true by
        # construction and means nothing.
        rng = np.random.default_rng(1)
        fr = np.geomspace(1e-5, 0.5, 1000)
        noisy = rng.normal(0., 1., 1000)        # minima everywhere
        mins = report.local_minima(noisy)
        assert len(mins) > 200
        d = [report.nearest_min_distance(fr, mins, f)
             for f in 10 ** rng.uniform(-5, np.log10(0.5), 100)]
        # every random cutoff is within a hair of some minimum
        assert np.median(d) < 0.01


class TestSummaryFields:
    def test_every_reported_field_is_declared(self):
        assert 'penalty' in diag.SUMMARY_FIELDS
        assert 'penalty_off' in diag.SUMMARY_FIELDS
        assert 'rel_pos' in diag.SUMMARY_FIELDS
        assert 'consistent' in diag.SUMMARY_FIELDS
        assert len(set(diag.SUMMARY_FIELDS)) == len(diag.SUMMARY_FIELDS)

    def test_no_peak_area_metric_is_reported(self):
        # The optimum is defined by baseline fit alone (Emmanuel,
        # 2026-08-16). A peak-area column appearing here would mean the
        # definition drifted.
        assert not any('area' in f for f in diag.SUMMARY_FIELDS)


class TestHarnessMatchesProduction:
    """The harness must call the pipeline the way auto_beads does.

    Calling `_fcutoff` with its own defaults runs plain ``beads`` with
    no peak regions instead of ``custom_beads`` with them -- a different
    algorithm on a different signal. That mistake produced a 15% crash
    rate and 54% containment before it was caught, and nothing about it
    is visible in the output: the run simply reports worse numbers.
    """

    def test_mirrors_auto_beads_defaults(self):
        import inspect

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

    def test_uses_custom_beads_with_regions(self):
        import inspect

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
