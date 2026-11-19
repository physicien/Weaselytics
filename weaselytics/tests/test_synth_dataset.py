"""Tests for the synthetic-benchmark generator in ``tools/``.

The module is loaded by path from ``tools/synthetic/``. These tests
exist because the generator defines what "correct baseline" means for
every constant grounded against it: an error here makes a wrong
*target*.
"""

import argparse
import datetime
import importlib.util
import pathlib
import re
import shlex
import subprocess
import sys

import numpy as np
import pytest

_PATH = (pathlib.Path(__file__).resolve().parents[2]
         / "tools" / "synthetic" / "synth_dataset.py")
_spec = importlib.util.spec_from_file_location("synth_dataset", _PATH)
synth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synth)


class TestQuantise:
    def test_values_land_on_the_lattice(self):
        rng = np.random.default_rng(4)
        y = rng.normal(0., 1., 5000)
        yq = synth.quantise(y)
        ratio = yq / synth.ADC_STEP_MV
        np.testing.assert_allclose(ratio, np.round(ratio), atol=1e-9)

    def test_error_at_most_half_a_step(self):
        rng = np.random.default_rng(5)
        y = rng.normal(0., 1., 5000)
        err = np.abs(synth.quantise(y) - y)
        assert err.max() <= synth.ADC_STEP_MV / 2 + 1e-12

    def test_shows_the_lattice_signature(self):
        # Quantized white noise must show the signature of a digitized
        # signal: consecutive differences that are exact multiples of the
        # step, and many exact ties.
        rng = np.random.default_rng(6)
        y = synth.quantise(rng.normal(0., 0.012, 20000))
        d = np.diff(y)
        ratio = d / synth.ADC_STEP_MV
        np.testing.assert_allclose(ratio, np.round(ratio), atol=1e-9)
        assert np.mean(d == 0) > 0.1

    def test_rejects_bad_step(self):
        with pytest.raises(ValueError, match="step"):
            synth.quantise(np.zeros(10), step=0.0)


class TestDeadTimeArtifact:
    @pytest.mark.parametrize("ratio", [0.05, 0.5, 1.0, 1.9])
    def test_trace_shows_the_drawn_ratio(self, ratio):
        t = np.arange(1200) / 120.
        pos = synth.emg_peak(t, 4.5, 0.03, 0.015, 2.)
        neg = synth.emg_peak(t, 4.54, 0.06, 0.03, 1.)
        c = synth._dip_scale(pos, neg, ratio)
        y = pos - c * neg
        assert -y.min() / y.max() == pytest.approx(ratio, rel=1e-6)

    def test_trace_shows_the_recorded_ratio(self):
        t = np.arange(1200) / 120.
        for seed in range(50):
            y, entries = synth.dead_time_artifact(
                t, 4.5, t[1] - t[0], np.random.default_rng(seed))
            ratio = entries[1]['dip_ratio']
            assert 0. <= ratio <= 2.
            assert -min(y.min(), 0.) / y.max() == pytest.approx(
                ratio, rel=1e-6, abs=1e-9)

    def test_rejects_an_empty_dip_lobe(self):
        pos = synth.emg_peak(np.arange(100) / 120., 0.4, 0.03, 0.015, 2.)
        with pytest.raises(ValueError, match="dip lobe"):
            synth._dip_scale(pos, np.zeros(100), 0.5)


class TestErbBaselines:
    def test_parabola_ends_at_zero_on_both_ends(self):
        # kind 0 is described in the source as "simple parabola that
        # ends at 0 on both ends"; that is what makes it the benign
        # case for the BEADS periodicity requirement.
        x = np.linspace(0., 1000., 1000)
        b = synth.erb_baseline(x, 0)
        assert b[0] == pytest.approx(0., abs=1e-9)
        assert b[-1] == pytest.approx(0., abs=1e-9)

    def test_exponential_starts_at_zero(self):
        x = np.linspace(0., 1000., 1000)
        assert synth.erb_baseline(x, 1)[0] == pytest.approx(0., abs=1e-12)

    def test_sinusoid_stays_within_its_amplitude(self):
        x = np.linspace(0., 1000., 1000)
        b = synth.erb_baseline(x, 3)
        assert b.min() >= 9. - 1e-9 and b.max() <= 11. + 1e-9

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            synth.erb_baseline(np.linspace(0., 1., 10), 7)

    def test_bare_cumsum_scales_with_sampling_density(self):
        # Erb's expression is a Riemann sum missing its dx, so
        # at fixed span its amplitude grows with the point count. This
        # is why the population uses exact_integral instead; the test
        # pins the defect so it cannot be "fixed" silently in the fixed
        # cases, where it is what the source does.
        def rng_of(n, exact):
            x = np.linspace(0., 1000., n)
            b = synth.erb_baseline(x, 2, exact_integral=exact)
            return b.max() - b.min()

        assert rng_of(2000, False) > 1.8 * rng_of(1000, False)
        # ... and does not, once the dx is restored
        assert rng_of(2000, True) == pytest.approx(rng_of(1000, True),
                                                   rel=0.02)


class TestErbSignal:
    def test_components_sum_to_the_signal_before_quantisation(self):
        rng = np.random.default_rng(0)
        d = synth.erb_signal(1800, 'many', 2, 0.019, rng,
                             quantise_output=False)
        np.testing.assert_allclose(
            d['y'], d['signal'] + d['baseline'] + d['noise'], rtol=0,
            atol=1e-12)

    def test_quantised_output_lands_on_the_detector_lattice(self):
        rng = np.random.default_rng(0)
        d = synth.erb_signal(900, 'many', 1, 0.019, rng)
        ratio = d['y'] / synth.ADC_STEP_MV
        np.testing.assert_allclose(ratio, np.round(ratio), atol=1e-9)

    def test_reproducible_for_a_given_seed(self):
        a = synth.erb_signal(900, 'isocratic', 0, 0.019,
                             np.random.default_rng(3))
        b = synth.erb_signal(900, 'isocratic', 0, 0.019,
                             np.random.default_rng(3))
        np.testing.assert_array_equal(a['y'], b['y'])

    def test_signal_is_the_shared_peak_component(self):
        # The peaks of `erb` are the peak component's own: same seed,
        # same draws.
        n, case = 1800, 'many'
        t1, sig1, pk1, t01, _ = synth.peak_component(
            n, case, np.random.default_rng(11), 0.02)
        d = synth.erb_signal(n, case, 2, 0.02,
                             np.random.default_rng(11))
        np.testing.assert_array_equal(d['signal'], sig1)
        np.testing.assert_array_equal(d['x'], t1)
        assert d['meta']['dead_time'] == t01
        assert len(d['peaks']) == len(pk1)

    def test_baseline_is_erbs_and_excludes_the_artefact(self):
        rng = np.random.default_rng(5)
        n = 1800
        d = synth.erb_signal(n, 'blank', 3, 0.019, rng)
        u = np.linspace(0., 1000., n)
        np.testing.assert_array_equal(
            d['baseline'], synth.erb_baseline(u, 3, exact_integral=True))
        # the injection artifact is in the signal, never in the truth
        art = [p for p in d['peaks'] if p.get('artifact')]
        assert art, "the injection artefact must be present"
        assert d['signal'].min() < 0, "its negative lobe must survive"

    def test_baseline_shape_does_not_depend_on_record_length(self):
        # erb_baseline is evaluated on a normalized abscissa, so the
        # four types stay comparable across the record lengths.
        rngs = [np.random.default_rng(1) for _ in range(2)]
        a = synth.erb_signal(900, 'blank', 2, 0.019, rngs[0])
        b = synth.erb_signal(4000, 'blank', 2, 0.019, rngs[1])
        assert (a['meta']['baseline_range']
                == pytest.approx(b['meta']['baseline_range'], rel=0.02))

    def test_all_four_baselines_and_every_peak_case_run(self):
        for case in synth.PEAK_CASES:
            for bt in range(4):
                d = synth.erb_signal(900, case, bt, 0.019,
                                     np.random.default_rng(0))
                assert np.all(np.isfinite(d['y']))
                lo, hi = synth.PEAK_CASES[case]
                assert lo <= d['meta']['n_requested'] <= hi
                assert (d['meta']['n_analytes'] + d['meta']['n_dropped']
                        == d['meta']['n_requested'])

    def test_sampling_is_the_measured_rate(self):
        d = synth.erb_signal(1200, 'blank', 0, 0.019,
                             np.random.default_rng(0))
        dt = np.median(np.diff(d['x']))
        assert dt == pytest.approx(1. / synth.PTS_PER_MIN, rel=1e-12)

    def test_a_blank_holds_only_the_injection_artifact(self):
        for bt in range(3):
            d = synth.erb_signal(1500, 'blank', bt, 0.02,
                                 np.random.default_rng(bt))
            assert len(d['peaks']) == 2
            assert all(p['artifact'] for p in d['peaks'])

    @pytest.mark.parametrize('model', synth.PEAK_MODELS)
    def test_nothing_elutes_before_the_dead_time(self, model):
        # The dead time is the transit time of an unretained species, so
        # it bounds every retention time in the run below, ghosts and
        # carryover included.
        shapes = dict(synth.DEFAULT_SHAPES, model=model)
        for k, case in enumerate(synth.PEAK_CASES):
            for bt in range(3):
                rng = np.random.default_rng(100 * k + bt)
                n = int(rng.integers(450, 1400))
                d = synth.erb_signal(n, case, bt, 0.05, rng, shapes=shapes)
                t0 = d['meta']['dead_time']
                assert all(p['tc'] >= t0 for p in d['peaks'][2:]), (
                    f'{case}/{bt} n={n}: a peak elutes before the '
                    f'dead time')

    @pytest.mark.parametrize('model', synth.PEAK_MODELS)
    def test_no_peak_overlaps_the_dead_time(self, model):
        # Not even its front tail: every peak is within the noise until
        # the injection artifact has returned to it.
        noise = 0.02
        shapes = dict(synth.DEFAULT_SHAPES, model=model)
        for k, case in enumerate(synth.PEAK_CASES):
            for bt in range(3):
                rng = np.random.default_rng(300 * k + bt)
                n = int(rng.integers(500, 3000))
                d = synth.erb_signal(n, case, bt, noise, rng, shapes=shapes)
                t = d['x']
                pos, neg = d['peaks'][:2]
                art = (synth.emg_peak(t, pos['tc'], pos['sigma'],
                                      pos['tau'], pos['height'])
                       - synth.emg_peak(t, neg['tc'], neg['sigma'],
                                        neg['tau'], -neg['height']))
                front = t <= t[np.nonzero(np.abs(art) > noise)[0][-1]]
                for p in d['peaks'][2:]:
                    y = synth.analyte_peak(t, p['tc'], p['sigma'],
                                           p['shape'], p['height'])
                    assert y[front].max() <= noise, (
                        f'{case}/{bt} n={n}: a peak overlaps the '
                        f'dead time')

    @pytest.mark.parametrize('model', synth.PEAK_MODELS)
    def test_the_record_ends_in_the_noise(self, model):
        # A peak has eluted when its trailing edge is back in the noise.
        noise = 0.02
        shapes = dict(synth.DEFAULT_SHAPES, model=model)
        for k, case in enumerate(synth.PEAK_CASES):
            for bt in range(3):
                rng = np.random.default_rng(7000 + 100 * k + bt)
                n = int(rng.integers(500, 3000))
                d = synth.erb_signal(n, case, bt, noise, rng, shapes=shapes)
                for p in d['peaks'][2:]:
                    y = synth.analyte_peak(d['x'], p['tc'], p['sigma'],
                                           p['shape'], p['height'])
                    assert y[-1] <= noise, (
                        f'{case}/{bt} n={n}: the record ends part-way '
                        f'down a peak')

    @pytest.mark.parametrize('model', synth.PEAK_MODELS)
    def test_adjacent_analytes_are_resolved(self, model):
        shapes = dict(synth.DEFAULT_SHAPES, model=model)
        for k, case in enumerate(('few', 'many', 'isocratic')):
            for seed in range(6):
                rng = np.random.default_rng(50 * k + seed)
                n = int(rng.integers(500, 3000))
                d = synth.erb_signal(n, case, seed % 3, 0.02, rng,
                                     shapes=shapes)
                rs = d['meta']['rs_min']
                if d['meta']['n_analytes'] >= 2:
                    assert rs >= synth.MIN_RESOLUTION - 1e-9, (
                        f'{case} n={n}: Rs {rs:.3f}')

    def test_true_apex_and_width_are_the_peaks_own(self):
        # `tc` and `fwhm_points` describe the Gaussian component of the
        # EMG; `apex` and `fwhm_true_points` must describe the peak.
        d = synth.erb_signal(2000, 'few', 0, 0.02,
                             np.random.default_rng(4))
        dt = d['x'][1] - d['x'][0]
        fine = np.linspace(d['x'][0], d['x'][-1], 40 * len(d['x']))
        for p in d['peaks'][2:]:
            y = synth.emg_peak(fine, p['tc'], p['sigma'], p['tau'],
                               p['height'])
            above = fine[y >= 0.5 * y.max()]
            assert p['apex'] == pytest.approx(fine[np.argmax(y)],
                                              abs=0.5 * dt)
            assert p['fwhm_true_points'] == pytest.approx(
                (above[-1] - above[0]) / dt, abs=1.)
            assert p['apex'] > p['tc']

    def test_the_dead_time_moves_earlier_to_make_room(self):
        # A 900-point record lasts 7.5 min: six to eight analytes fit
        # only once the dead time has been drawn again, earlier.
        d = synth.erb_signal(900, 'many', 0, 0.02,
                             np.random.default_rng(1))
        assert d['meta']['dead_time_redraws'] > 0
        assert d['meta']['n_dropped'] == 0
        assert 2.5 <= d['meta']['dead_time'] <= 6.5

    def test_a_blank_keeps_its_first_dead_time(self):
        for seed in range(10):
            d = synth.erb_signal(600, 'blank', 0, 0.02,
                                 np.random.default_rng(seed))
            assert d['meta']['dead_time_redraws'] == 0

    def test_peaks_that_fit_nowhere_are_dropped(self):
        # A 500-point record lasts about 4 min: even with the dead time
        # at 2.5 min, not every one of six to eight analytes fits. Those
        # that cannot are dropped rather than left running off the end.
        noise = 0.02
        d = synth.erb_signal(500, 'many', 0, noise,
                             np.random.default_rng(0))
        assert d['meta']['n_dropped'] > 0
        assert d['meta']['dead_time'] - 2.5 < 1. / synth.PTS_PER_MIN
        assert (d['meta']['n_analytes'] + d['meta']['n_dropped']
                == d['meta']['n_requested'])
        assert d['signal'][-1] <= noise

    def test_dead_time_is_drawn_inside_its_range(self):
        for k in range(50):
            rng = np.random.default_rng(k)
            n = int(rng.integers(500, 4000))
            d = synth.erb_signal(n, 'blank', 0, 0.02, rng)
            t0 = d['meta']['dead_time']
            assert 2.5 <= t0 <= min(6.5, 0.88 * d['meta']['minutes'])

    def test_a_record_too_short_for_the_dead_time_is_refused(self):
        with pytest.raises(ValueError, match='dead time'):
            synth.peak_component(300, 'blank',
                                 np.random.default_rng(0), 0.02)

    def test_record_durations_stay_inside_the_range(self):
        rng = np.random.default_rng(0)
        lo, hi = synth.ERB_N_RANGE
        d = [synth.draw_record_duration(rng) for _ in range(2000)]
        n = np.round(np.array(d) * synth.PTS_PER_MIN)
        assert lo <= n.min() and n.max() <= hi


class TestPeakShapes:
    """The analyte models, their tailing factor and the defaults."""

    _X = np.linspace(-60., 400., 400001)

    def _measured(self, shape):
        return synth.tailing_factor(
            self._X, synth.analyte_peak(self._X, 0., 1., shape, 1.))

    @pytest.mark.parametrize('tf', [1.05, 1.5, 3.0])
    def test_emg_reaches_the_asked_tailing_factor(self, tf):
        shape = synth.shape_for_tf('emg', tf)
        assert self._measured(shape) == pytest.approx(tf, abs=1e-4)

    @pytest.mark.parametrize('m', [3.5, 51.])
    @pytest.mark.parametrize('tf', [0.76, 0.9, 1.2, 2.0])
    def test_pearson7_reaches_the_asked_tailing_factor(self, m, tf):
        shape = synth.shape_for_tf('pearson7', tf, m)
        assert self._measured(shape) == pytest.approx(tf, abs=1e-4)
        assert (shape['e'] < 0) == (tf < 1)

    def test_emg_cannot_front(self):
        with pytest.raises(ValueError, match='front'):
            synth.shape_for_tf('emg', 0.9)

    def test_gaussian_is_symmetric(self):
        assert self._measured({'model': 'gaussian'}) == pytest.approx(
            1., abs=1e-6)

    @pytest.mark.parametrize('m', [1.5, 3.5, 51.])
    def test_symmetric_pearson7_has_the_plate_fwhm(self, m):
        y = synth.analyte_peak(self._X, 0., 1.,
                               {'model': 'pearson7', 'm': m, 'e': 0.}, 1.)
        _, left, right = synth._crossings(self._X, y, 0.5)
        assert right - left == pytest.approx(synth._FWHM_PER_SIGMA,
                                             rel=1e-6)

    @pytest.mark.parametrize('e', [0.28, -0.28])
    def test_pearson7_long_side_falls_below_its_floor(self, e):
        # Without its floor the long side keeps falling, as the inverse
        # of the distance, where the fitted function levels off.
        shape = {'model': 'pearson7', 'm': 3.5, 'e': e}
        floor = synth.pearson7_floor(3.5, e)
        assert floor > 1e-3
        x = np.sign(e) * np.array([100., 1000., 10000.])
        far = synth.analyte_peak(x, 0., 1., shape, 1.)
        assert far[0] < floor
        assert far[0] > far[1] > far[2] > 0.
        assert far[1] * 1000. == pytest.approx(far[2] * 10000., rel=0.05)
        y = synth.analyte_peak(self._X, 0., 1., shape, 1.)
        assert y.max() == pytest.approx(1., abs=1e-6)

    def test_default_draws_are_reproducible(self):
        # The seed alone fixes every shape; the values are pinned so a
        # change in the order of the draws shows.
        d = synth.erb_signal(1800, 'many', 0, 0.02,
                             np.random.default_rng(7))
        analytes = d['peaks'][2:]
        np.testing.assert_allclose(
            [p['tailing_factor'] for p in analytes],
            [1.14154134, 1.03148145, 1.25685575, 1.14185173, 1.29424901,
             1.16019759, 1.14555593, 1.01872069], rtol=1e-7)
        np.testing.assert_allclose(
            [p['height'] for p in analytes],
            [5.6626906987, 1.724457917, 1.6910550507, 5.4192890078,
             3.5143898728, 1.1290264026, 1.9239832923, 22.6343492042],
            rtol=1e-9)
        for p in analytes:
            assert p['shape']['model'] == 'emg'
            assert p['tau'] == pytest.approx(
                p['shape']['tau_ratio'] * p['sigma'])

    @pytest.mark.parametrize('model', ['emg', 'pearson7'])
    def test_default_tailing_follows_its_distribution(self, model):
        # One plus a log-normal of median TF_EXCESS_MEDIAN; only a
        # Pearson VII fronts, at FRONTING_SHARE.
        rng = np.random.default_rng(1)
        shapes = dict(synth.DEFAULT_SHAPES, model=model)
        drawn = [synth.draw_shape(rng, shapes) for _ in range(400)]
        tf = np.array([synth.shape_tailing_factor(s) for s in drawn])
        assert np.median(tf) == pytest.approx(
            1. + synth.TF_EXCESS_MEDIAN, abs=0.03)
        assert np.std(np.log(tf[tf > 1.] - 1.)) == pytest.approx(
            synth.TF_EXCESS_LOG_SIGMA, abs=0.1)
        fronting = np.mean(tf < 1.)
        if model == 'emg':
            assert fronting == 0.
        else:
            assert 0.005 < fronting < 0.04
            log_m = np.log10([s['m'] for s in drawn])
            lo, hi = np.log10(synth.PEARSON7_M_RANGE)
            assert lo <= log_m.min() and log_m.max() <= hi
            assert log_m.mean() == pytest.approx((lo + hi) / 2., abs=0.06)

    def test_emg_factors_match_independent_values(self):
        # tau/sigma = 1.2, computed apart from the generator.
        tf, a_s = synth.shape_factors({'model': 'emg', 'tau_ratio': 1.2})
        assert tf == pytest.approx(1.3095, abs=1e-4)
        assert a_s == pytest.approx(1.4935, abs=1e-4)
        y = synth.analyte_peak(self._X, 0., 1.,
                               {'model': 'emg', 'tau_ratio': 1.2}, 1.)
        assert synth.asymmetry_factor(self._X, y) == pytest.approx(
            a_s, abs=1e-4)

    @pytest.mark.parametrize('e', [-0.2, 0., 0.25])
    def test_pearson7_apex_is_at_its_centre(self, e):
        y = synth.analyte_peak(self._X, 3., 1.,
                               {'model': 'pearson7', 'm': 3.5, 'e': e}, 2.)
        apex, _, _ = synth._crossings(self._X, y, 0.5)
        assert apex == pytest.approx(3., abs=1e-3)
        assert y.max() == pytest.approx(2., rel=1e-6)

    @pytest.mark.parametrize('shape', [
        {'model': 'pearson7', 'm': 3.5, 'e': 0.2},
        {'model': 'gaussian'},
    ])
    def test_profiles_match_one_peak_at_a_time(self, shape):
        t = np.linspace(0., 20., 2001)
        centres, sigmas = np.array([5., 9.]), np.array([0.3, 0.6])
        rows = synth._profiles(t, centres, sigmas, shape)
        for row, c, s in zip(rows, centres, sigmas):
            np.testing.assert_allclose(
                row / row.max(), synth.analyte_peak(t, c, s, shape, 1.),
                atol=1e-12)

    def test_a_range_out_of_reach_raises(self):
        shapes = dict(synth.DEFAULT_SHAPES, tailing_factor=(0.5, 0.9))
        with pytest.raises(ValueError, match='front'):
            synth.draw_shape(np.random.default_rng(0), shapes)

    @pytest.mark.parametrize('tf', [0.3, 40.])
    def test_pearson7_refuses_a_tailing_factor_out_of_reach(self, tf):
        with pytest.raises(ValueError, match='out of reach'):
            synth.shape_for_tf('pearson7', tf, 3.5)

    @pytest.mark.parametrize('model, tf', [
        ('emg', (1.3, 1.6)), ('pearson7', (0.8, 1.5)), ('gaussian', None),
    ])
    def test_recorded_tailing_factors_follow_the_asked_range(self, model,
                                                             tf):
        shapes = dict(synth.DEFAULT_SHAPES, model=model, tailing_factor=tf)
        d = synth.erb_signal(6000, 'many', 0, 0.02,
                             np.random.default_rng(2), shapes=shapes)
        analytes = d['peaks'][2:]
        assert analytes
        lo, hi = tf if tf else (1., 1.)
        for p in analytes:
            assert p['shape']['model'] == model
            assert lo - 1e-6 <= p['tailing_factor'] <= hi + 1e-6
        assert d['meta']['peak_model'] == model

    @pytest.mark.parametrize('args', [
        ['--peak-model', 'emg', '--tailing-factor', '0.9', '1.2'],
        ['--peak-model', 'gaussian', '--tailing-factor', '1', '1.2'],
        ['--tailing-factor', '1.5', '1.2'],
        ['--pearson-m', '3', '10'],
        ['--peak-model', 'pearson7', '--tailing-factor', '0.3', '0.6'],
    ])
    def test_the_parser_refuses_impossible_shapes(self, args, tmp_path,
                                                  monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['synth_dataset.py',
                                          str(tmp_path / 'SET')] + args)
        with pytest.raises(SystemExit):
            synth.main()
        assert not (tmp_path / 'SET').exists()


class TestRate:
    def test_default_rate_is_120(self):
        a = synth.erb_signal(1500, 'few', 0, 0.02, np.random.default_rng(5))
        b = synth.erb_signal(1500, 'few', 0, 0.02, np.random.default_rng(5),
                             rate=120.)
        assert synth.PTS_PER_MIN == 120
        np.testing.assert_array_equal(a['y'], b['y'])

    def test_widths_in_minutes_do_not_depend_on_the_rate(self):
        t1 = np.arange(1200) / 120.
        t2 = np.arange(2400) / 240.
        _, e1 = synth.dead_time_artifact(t1, 4.5, t1[1] - t1[0],
                                         np.random.default_rng(3))
        _, e2 = synth.dead_time_artifact(t2, 4.5, t2[1] - t2[0],
                                         np.random.default_rng(3))
        for a, b in zip(e1, e2):
            assert b['sigma'] == pytest.approx(a['sigma'], rel=1e-12)
            assert b['fwhm_points'] == pytest.approx(2. * a['fwhm_points'],
                                                     rel=1e-12)

    def test_a_faster_rate_samples_the_same_run_more_densely(self):
        d = synth.erb_signal(3000, 'single', 1, 0.02,
                             np.random.default_rng(4), rate=240.)
        assert d['meta']['points_per_minute'] == 240.
        assert d['x'][1] - d['x'][0] == pytest.approx(1. / 240.)
        assert d['meta']['minutes'] == pytest.approx(2999. / 240.)


class TestNamesAndManifest:
    def test_stem_puts_the_padded_replicate_before_the_length(self):
        stem = synth.signal_stem('erb', 'many', 'step', 'typ', 3, 1874, 2)
        assert stem == 'SYN__erb__many__step__typ__03__1874'

    def test_padding_follows_the_widest_replicate(self):
        assert synth.signal_stem('erb', 'few', 'exp', 'high', 3, 900,
                                 1).endswith('__3__900')

    def test_manifest_row_matches_its_columns(self):
        d = synth.erb_signal(900, 'few', 0, 0.02, np.random.default_rng(1))
        stem = synth.signal_stem('erb', 'few', 'parabola', 'typ', 0, 900, 2)
        row = synth.manifest_row(stem, 'erb', 'few', 'parabola', 'typ',
                                 900, 0, d).split(',')
        assert len(row) == len(synth.MANIFEST_COLUMNS)
        fields = dict(zip(synth.MANIFEST_COLUMNS, row))
        # the artifact lobes are not analytes
        assert int(fields['n_analytes']) == d['meta']['n_analytes']
        assert int(fields['n_analytes']) == len(d['peaks']) - 2
        assert float(fields['points_per_minute']) == synth.PTS_PER_MIN


class TestProvenance:
    def _write(self, tmp_path, **overrides):
        params = dict(output='SET', seed=0, replicates=1,
                      rate=synth.PTS_PER_MIN, no_plot=False)
        params.update(overrides)
        path = tmp_path / 'PROVENANCE.txt'
        started = datetime.datetime.now().astimezone()
        synth.write_provenance(str(path), argparse.Namespace(**params),
                               ['synth_dataset.py', 'SET'], started,
                               'COMMIT')
        return path.read_text().splitlines()

    def test_writes_every_line(self, tmp_path):
        lines = self._write(tmp_path)
        assert [ln.split()[0] for ln in lines] == [
            'commit', 'command', 'arguments', 'machine', 'python']

    def test_records_the_commit_it_is_given(self, tmp_path):
        assert self._write(tmp_path)[0] == 'commit COMMIT'

    def test_records_the_exact_rate(self, tmp_path):
        line = self._write(tmp_path, rate=1234.5678)[2]
        assert 'rate=1234.5678' in line.split()

    def test_records_the_defaults(self, tmp_path):
        args = self._write(tmp_path)[2].split()[1:]
        assert 'seed=0' in args
        assert 'replicates=1' in args
        assert 'rate=120' in args
        assert 'plot=True' in args
        output = next(a for a in args if a.startswith('output='))
        assert pathlib.Path(output.split('=', 1)[1]).is_absolute()

    def test_records_no_plot(self, tmp_path):
        assert 'plot=False' in self._write(tmp_path, no_plot=True)[2]

    def test_command_starts_with_the_interpreter(self, tmp_path):
        command = self._write(tmp_path)[1]
        expected = shlex.join([sys.executable, 'synth_dataset.py', 'SET'])
        assert command == f'command {expected}'

    def test_commit_is_a_full_hash(self):
        commit = synth.git_commit()
        if commit == 'unknown':
            pytest.skip('not run from a git checkout')
        # SHA-1 or SHA-256 repositories
        assert re.fullmatch(r'([0-9a-f]{40}|[0-9a-f]{64})(-dirty|-unknown)?',
                            commit)

    @pytest.mark.parametrize('code, suffix', [
        (0, ''), (1, '-dirty'), (128, '-unknown'),
    ])
    def test_tree_state_from_the_diff_exit_code(self, monkeypatch, code,
                                                suffix):
        def run(cmd, **kwargs):
            if 'rev-parse' in cmd:
                return subprocess.CompletedProcess(cmd, 0, 'a' * 40 + '\n')
            return subprocess.CompletedProcess(cmd, code)
        monkeypatch.setattr(synth.subprocess, 'run', run)
        assert synth.git_commit() == 'a' * 40 + suffix

    @pytest.mark.parametrize('error', [
        OSError('git not found'),
        subprocess.CalledProcessError(128, 'git'),
    ])
    def test_commit_unknown_without_git(self, monkeypatch, error):
        def fail(*args, **kwargs):
            raise error
        monkeypatch.setattr(synth.subprocess, 'run', fail)
        assert synth.git_commit() == 'unknown'


class TestMain:
    def test_writes_provenance_after_the_manifest(self, tmp_path,
                                                  monkeypatch):
        out = tmp_path / 'SET'
        monkeypatch.setattr(sys, 'argv', [
            'synth_dataset.py', str(out), '--replicates', '1', '--no-plot'])
        synth.main()
        manifest = out / 'manifest.csv'
        provenance = out / 'PROVENANCE.txt'
        assert provenance.stat().st_mtime_ns >= manifest.stat().st_mtime_ns
        n_rows = len(manifest.read_text().splitlines()) - 1
        assert n_rows == len(list((out / 'signals').glob('*.txt')))
        assert 'replicates=1' in provenance.read_text()

    def test_refuses_a_non_empty_directory(self, tmp_path, monkeypatch):
        (tmp_path / 'old.txt').write_text('x')
        monkeypatch.setattr(sys, 'argv', ['synth_dataset.py', str(tmp_path)])
        with pytest.raises(SystemExit):
            synth.main()
        assert sorted(p.name for p in tmp_path.iterdir()) == ['old.txt']
