"""Tests for the synthetic-benchmark generator in ``tools/``.

The module is loaded by path from ``tools/synthetic/``. These tests
exist because the generator defines what "correct baseline" means for
every constant grounded against it: an error here makes a wrong
*target*.
"""

import importlib.util
import pathlib

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

    def test_peak_component_matches_the_native_family_exactly(self):
        # The peaks of `erb` are native's own: same seed, same
        # draws.
        n, case = 1800, 'many'
        t1, sig1, pk1, t01, _ = synth.native_peak_component(
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

    def test_nothing_elutes_before_the_dead_time(self):
        # The dead time is the transit time of an unretained species, so
        # it bounds every retention time in the run below, ghosts and
        # carryover included.
        for k, case in enumerate(synth.PEAK_CASES):
            for bt in range(3):
                rng = np.random.default_rng(100 * k + bt)
                n = int(rng.integers(450, 1400))
                d = synth.erb_signal(n, case, bt, 0.05, rng)
                t0 = d['meta']['dead_time']
                assert all(p['tc'] >= t0 for p in d['peaks'][2:]), (
                    f'{case}/{bt} n={n}: a peak elutes before the '
                    f'dead time')

    def test_no_peak_overlaps_the_dead_time(self):
        # Not even its front tail: every peak is within the noise until
        # the injection artifact has returned to it.
        noise = 0.02
        for k, case in enumerate(synth.PEAK_CASES):
            for bt in range(3):
                rng = np.random.default_rng(300 * k + bt)
                n = int(rng.integers(500, 3000))
                d = synth.erb_signal(n, case, bt, noise, rng)
                t = d['x']
                pos, neg = d['peaks'][:2]
                art = (synth.emg_peak(t, pos['tc'], pos['sigma'],
                                      pos['tau'], pos['height'])
                       - synth.emg_peak(t, neg['tc'], neg['sigma'],
                                        neg['tau'], -neg['height']))
                front = t <= t[np.nonzero(np.abs(art) > noise)[0][-1]]
                for p in d['peaks'][2:]:
                    y = synth.emg_peak(t, p['tc'], p['sigma'], p['tau'],
                                       p['height'])
                    assert y[front].max() <= noise, (
                        f'{case}/{bt} n={n}: a peak overlaps the '
                        f'dead time')

    def test_the_record_ends_in_the_noise(self):
        # A peak has eluted when its trailing edge is back in the noise.
        noise = 0.02
        for k, case in enumerate(synth.PEAK_CASES):
            for bt in range(3):
                rng = np.random.default_rng(7000 + 100 * k + bt)
                n = int(rng.integers(500, 3000))
                d = synth.erb_signal(n, case, bt, noise, rng)
                for p in d['peaks'][2:]:
                    y = synth.emg_peak(d['x'], p['tc'], p['sigma'],
                                       p['tau'], p['height'])
                    assert y[-1] <= noise, (
                        f'{case}/{bt} n={n}: the record ends part-way '
                        f'down a peak')

    def test_adjacent_analytes_are_resolved(self):
        for k, case in enumerate(('few', 'many', 'isocratic')):
            for seed in range(6):
                rng = np.random.default_rng(50 * k + seed)
                n = int(rng.integers(500, 3000))
                d = synth.erb_signal(n, case, seed % 3, 0.02, rng)
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
            synth.native_peak_component(300, 'blank',
                                        np.random.default_rng(0), 0.02)

    def test_record_lengths_stay_inside_the_range(self):
        rng = np.random.default_rng(0)
        lo, hi = synth.ERB_N_RANGE
        n = [synth.draw_record_length(rng) for _ in range(2000)]
        assert lo <= min(n) and max(n) <= hi


class TestNativeSignal:
    def test_reports_its_placement_like_erb(self):
        d = synth.make_signal(900, 'many', 'exp', 0.02,
                              np.random.default_rng(2))
        m = d['meta']
        assert m['family'] == 'native'
        lo, hi = synth.PEAK_CASES['many']
        assert lo <= m['n_requested'] <= hi
        assert m['n_analytes'] + m['n_dropped'] == m['n_requested']
        assert m['n_analytes'] == sum(not p['artifact'] for p in d['peaks'])

    def test_a_blank_holds_only_the_injection_artifact(self):
        d = synth.make_signal(1800, 'blank', 'hump', 0.02,
                              np.random.default_rng(0))
        assert len(d['peaks']) == 2
        assert all(p['artifact'] for p in d['peaks'])


class TestNamesAndManifest:
    def test_stem_puts_the_padded_replicate_before_the_length(self):
        stem = synth.signal_stem('erb', 'many', 'step', 'typ', 3, 1874, 2)
        assert stem == 'SYN__erb__many__step__typ__03__1874'

    def test_padding_follows_the_widest_replicate(self):
        assert synth.signal_stem('native', 'few', 'exp', 'high', 3, 900,
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
