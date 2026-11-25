"""Tests for the synthetic-benchmark generator in ``tools/``.

The module is loaded by path from ``tools/synthetic/``. These tests
exist because the generator defines what "correct baseline" means for
every constant grounded against it: an error here makes a wrong
*target*.
"""

import argparse
import datetime
import importlib.util
import json
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


_GAUSSIAN = dict(synth.DEFAULT_SHAPES, model='gaussian')


def _layout(**changes):
    return dict(synth.DEFAULT_LAYOUT, **changes)


def _analytes(d):
    return [p for p in d['peaks'] if not p['artifact']]


def _profile(t, p):
    return synth.analyte_peak(t, p['tc'], p['sigma'], p['shape'],
                              p['height'])


class TestArtifact:
    @pytest.mark.parametrize("ratio", [0.05, 0.5, 1.0, 1.9])
    def test_trace_shows_the_drawn_ratio(self, ratio):
        t = np.arange(1200.)
        pos = synth.emg_peak(t, 540., 3.6, 1.8, 2.)
        neg = synth.emg_peak(t, 545., 7.2, 3.6, 1.)
        c = synth._dip_scale(pos, neg, ratio)
        y = pos - c * neg
        assert -y.min() / y.max() == pytest.approx(ratio, rel=1e-6)

    def test_trace_shows_the_recorded_ratio(self):
        t = np.arange(1200.)
        for seed in range(50):
            params = synth.draw_artifact(np.random.default_rng(seed), 3.4, 2.)
            y, entries = synth.artifact_profile(t, 540.3, params)
            ratio = entries[1]['dip_ratio']
            assert 0. <= ratio <= 2.
            assert -min(y.min(), 0.) / y.max() == pytest.approx(
                ratio, rel=1e-6, abs=1e-9)

    def test_the_artifact_is_as_wide_as_a_peak_at_its_place(self):
        for s in range(20):
            d = synth.make_signal(4000, 1, 0.02, np.random.default_rng(s),
                                  shapes=_GAUSSIAN,
                                  layout=_layout(plate_model=True))
            spike = [p for p in d['peaks'] if p['artifact']][0]
            plates = d['meta']['n_plates']
            jitter = spike['sigma'] * np.sqrt(plates) / spike['tc']
            assert 0.85 - 1e-9 <= jitter <= 1.15 + 1e-9
            hi = synth.HEIGHT_MAX[1]
            assert hi / synth.HEIGHT_SPAN <= spike['height'] <= hi

    def test_the_truth_records_what_the_trace_shows(self):
        t = np.arange(1200.)
        lower = 0
        for seed in range(50):
            params = synth.draw_artifact(np.random.default_rng(seed), 3.4,
                                         2.)
            y, (spike, dip) = synth.artifact_profile(t, 540.3, params)
            assert spike['trace_height'] == y.max()
            assert dip['trace_height'] == y.min()
            assert spike['height'] == 2.
            lower += spike['trace_height'] < 0.95 * spike['height']
        assert lower > 10

    def test_extent_holds_at_any_phase(self):
        noise = 0.02
        for seed in range(20):
            params = synth.draw_artifact(np.random.default_rng(seed), 3.4, 2.)
            lead, trail = synth.artifact_extent(params, noise)
            t = np.arange(2000.)
            for t0 in (300., 300.25, 300.5, 300.75):
                y, _ = synth.artifact_profile(t, t0, params)
                above = t[np.abs(y) > noise]
                assert above[0] >= t0 - lead
                assert above[-1] <= t0 + trail

    def test_rejects_an_empty_dip_lobe(self):
        pos = synth.emg_peak(np.arange(100.), 48., 3.6, 1.8, 2.)
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


class TestBaselineTypes:
    def test_types_3_to_5_are_erbs(self):
        n = 1800
        u = np.linspace(0., 1000., n)
        for bt in (3, 4, 5):
            b, info = synth.draw_baseline(n, bt, np.random.default_rng(0))
            np.testing.assert_array_equal(
                b, synth.erb_baseline(u, bt - 3, exact_integral=True))
            assert info == {}

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match='baseline type'):
            synth.draw_baseline(100, 6, np.random.default_rng(0))

    def test_type1_draws_inside_its_ranges(self):
        rng = np.random.default_rng(1)
        orders = set()
        for _ in range(300):
            b, info = synth.type1_baseline(2000, rng)
            orders.add(info['order'])
            lo, hi = synth.TYPE1_CYCLES_RANGE
            assert lo <= info['cycles'] <= hi
            assert b.min() == pytest.approx(0., abs=1e-12)
        assert orders == set(range(synth.TYPE1_ORDER_RANGE[0],
                                   synth.TYPE1_ORDER_RANGE[1] + 1))

    @pytest.mark.parametrize('kind', [1, 2])
    def test_the_baseline_stretches_with_the_record(self, kind):
        # The same draw on a record four times longer is the same curve,
        # sampled four times more densely.
        a, _ = synth.draw_baseline(1000, kind, np.random.default_rng(3))
        b, _ = synth.draw_baseline(4000, kind, np.random.default_rng(3))
        if kind == 1:
            np.testing.assert_allclose(b[::4], a, atol=0.02 * np.ptp(a))
        else:
            # Type 2 draws new white noise at each length, so only its
            # spectrum stretches: the same number of waves per record.
            def crossings(x):
                return np.sum(np.diff(np.sign(x - x.mean())) != 0)
            ca = np.mean([crossings(synth.type2_baseline(
                1000, np.random.default_rng(s))[0]) for s in range(200)])
            cb = np.mean([crossings(synth.type2_baseline(
                4000, np.random.default_rng(s))[0]) for s in range(200)])
            assert cb == pytest.approx(ca, rel=0.1)

    def test_type2_power_is_below_its_cutoff(self):
        n = 2000
        spectra = [np.abs(np.fft.rfft(synth.type2_baseline(
            n, np.random.default_rng(s))[0])) ** 2 for s in range(200)]
        p = np.mean(spectra, axis=0)
        cycles = np.arange(len(p))
        below = p[cycles <= synth.TYPE2_CYCLES].sum()
        assert below / p.sum() > 0.9

    def test_type2_does_not_end_where_it_starts(self):
        # A filter in the Fourier domain is periodic; the padding breaks
        # it, so the two ends differ as much as two random samples.
        gaps = [abs(b[-1] - b[0]) / b.std() for b, _ in (
            synth.type2_baseline(2000, np.random.default_rng(s))
            for s in range(300))]
        assert np.median(gaps) > 0.5

    def test_type2_std_is_inside_its_range(self):
        lo, hi = synth.TYPE2_STD_RANGE
        for s in range(50):
            b, _ = synth.type2_baseline(1500, np.random.default_rng(s))
            assert lo - 1e-9 <= b.std() <= hi + 1e-9


class TestNoise:
    @pytest.mark.parametrize('kind', synth.BASELINE_TYPES)
    @pytest.mark.parametrize('name, snr', synth.NOISE_CASES.items())
    def test_the_noise_reaches_the_asked_ratio(self, kind, name, snr):
        d = synth.make_signal(20000, kind, None, np.random.default_rng(1),
                              snr=snr)
        ratio = 10. * np.log10(np.sum(d['baseline'] ** 2)
                               / np.sum(d['noise'] ** 2))
        assert ratio == pytest.approx(snr, abs=0.2)
        assert name == f'snr{snr:g}'
        assert d['meta']['noise_sigma'] == pytest.approx(
            synth.noise_for_snr(d['baseline'], snr))

    def test_the_ratio_leaves_the_rest_alone(self):
        # The baseline has a stream of its own, so drawing it first
        # moves neither the peaks nor the noise.
        sigma = 0.03
        a = synth.make_signal(3000, 2, sigma, np.random.default_rng(4))
        b = synth.make_signal(
            3000, 2, None, np.random.default_rng(4),
            snr=20. * np.log10(np.sqrt(np.mean(a['baseline'] ** 2)) / sigma))
        np.testing.assert_array_equal(a['baseline'], b['baseline'])
        np.testing.assert_allclose(a['noise'], b['noise'], rtol=1e-9)
        np.testing.assert_allclose(a['signal'], b['signal'], rtol=1e-6)


class TestPeakComponent:
    def test_components_sum_to_the_signal(self):
        d = synth.make_signal(1800, 1, 0.019, np.random.default_rng(0))
        np.testing.assert_allclose(
            d['y'], d['signal'] + d['baseline'] + d['noise'], rtol=0,
            atol=1e-12)

    def test_reproducible_for_a_given_seed(self):
        a = synth.make_signal(900, 2, 0.019, np.random.default_rng(3),
                              layout=_layout(plate_model=True))
        b = synth.make_signal(900, 2, 0.019, np.random.default_rng(3),
                              layout=_layout(plate_model=True))
        np.testing.assert_array_equal(a['y'], b['y'])

    def test_the_axis_is_the_sample_index(self):
        d = synth.make_signal(700, 3, 0.02, np.random.default_rng(0))
        np.testing.assert_array_equal(d['x'], np.arange(700.))

    def test_no_artifact_by_default(self):
        for s in range(20):
            d = synth.make_signal(1000, 1, 0.02, np.random.default_rng(s))
            assert not any(p['artifact'] for p in d['peaks'])
            assert np.isnan(d['meta']['dead_time'])
            assert d['meta']['n_requested'] >= synth.N_PEAKS_RANGE[0]

    def test_the_artifact_is_the_first_feature(self):
        noise = 0.02
        for s in range(30):
            d = synth.make_signal(1500, 1, noise, np.random.default_rng(s),
                                  layout=_layout(plate_model=True))
            art = [p for p in d['peaks'] if p['artifact']]
            assert len(art) == 2
            assert art[0]['tc'] == d['meta']['dead_time']
            t = d['x']
            front = np.abs(d['signal'] - sum(
                (_profile(t, p) for p in _analytes(d)), np.zeros(len(t))))
            t_front = t[np.nonzero(front > noise)[0][-1]]
            for p in _analytes(d):
                assert _profile(t, p)[t <= t_front].max() <= noise

    def test_the_artifact_stays_inside_the_record(self):
        noise = 0.05
        for s in range(40):
            d = synth.make_signal(500, 1, noise, np.random.default_rng(s),
                                  layout=_layout(plate_model=True, n_peaks=0))
            assert abs(d['signal'][0]) <= noise
            assert abs(d['signal'][-1]) <= noise

    @pytest.mark.parametrize('model', synth.PEAK_MODELS)
    @pytest.mark.parametrize('layout', [dict(), dict(plate_model=True)])
    def test_the_margins_hold_no_feature(self, model, layout):
        # Every peak and the artifact are within the noise in the first
        # and the last BORDER_MARGIN samples.
        noise = 0.02
        m = synth.BORDER_MARGIN
        shapes = dict(synth.DEFAULT_SHAPES, model=model)
        for s in range(12):
            rng = np.random.default_rng(s)
            n = int(rng.integers(500, 3000))
            d = synth.make_signal(n, 1, noise, rng, shapes=shapes,
                                  layout=_layout(**layout))
            ys = [_profile(d['x'], p) for p in _analytes(d)]
            ys.append(d['signal'] - sum(ys, np.zeros(n)))
            for y in ys:
                assert np.abs(y[:m]).max() <= noise + 1e-12
                assert np.abs(y[-m:]).max() <= noise + 1e-12

    def test_the_dead_time_is_in_the_first_third(self):
        for s in range(60):
            d = synth.make_signal(600, 1, 0.02, np.random.default_rng(s),
                                  shapes=_GAUSSIAN,
                                  layout=_layout(plate_model=True))
            t0 = d['meta']['dead_time']
            assert synth.BORDER_MARGIN <= t0 < 600. * synth.DEAD_TIME_SHARE
            assert all(p['tc'] > t0 for p in _analytes(d))
            assert d['meta']['dropped_early'] >= 0

    def test_the_dead_time_is_picked_at_random_among_the_spots(self):
        # Not always the earliest: earlier spots are then discarded.
        early = [synth.make_signal(2000, 1, 0.02, np.random.default_rng(s),
                                   shapes=_GAUSSIAN,
                                   layout=_layout(plate_model=True, n_peaks=20)
                                   )['meta']['dropped_early']
                 for s in range(60)]
        assert min(early) == 0 and max(early) > 3

    def test_the_whole_artifact_clears_the_start_margin(self):
        # A wide artifact at an early spot reaches into the margin: the
        # spot is refused, not only its centre checked.
        n, noise, m = 500, 0.02, synth.BORDER_MARGIN
        for s in range(200):
            d = synth.make_signal(
                n, 1, noise, np.random.default_rng(s), shapes=_GAUSSIAN,
                layout=_layout(plate_model=True, n_peaks=0))
            assert np.abs(d['signal'][:m]).max() <= noise + 1e-12

    def test_a_blank_holds_only_the_artifact(self):
        d = synth.make_signal(1500, 1, 0.02, np.random.default_rng(0),
                              layout=_layout(plate_model=True, n_peaks=0))
        assert len(d['peaks']) == 2
        assert all(p['artifact'] for p in d['peaks'])

    def test_a_record_too_short_for_the_artifact_is_refused(self):
        with pytest.raises(ValueError, match='no dead time'):
            synth.make_signal(5, 1, 0.02, np.random.default_rng(0),
                              layout=_layout(plate_model=True))

    def test_counts_follow_their_range(self):
        lo, hi = synth.N_PEAKS_RANGE
        for plate, low in ((False, lo), (True, 0)):
            k = [synth.make_signal(600, 1, 0.02, np.random.default_rng(s),
                                   shapes=_GAUSSIAN,
                                   layout=_layout(plate_model=plate)
                                   )['meta']['n_requested']
                 for s in range(300)]
            assert min(k) == low and max(k) == hi

    def test_the_count_leaves_the_baseline_and_the_noise_alone(self):
        a = synth.make_signal(4000, 1, 0.02, np.random.default_rng(4))
        k = a['meta']['n_requested']
        b = synth.make_signal(4000, 1, 0.02, np.random.default_rng(4),
                              layout=_layout(n_peaks=k))
        np.testing.assert_array_equal(a['y'], b['y'])
        c = synth.make_signal(4000, 1, 0.02, np.random.default_rng(4),
                              layout=_layout(n_peaks=k - 3))
        np.testing.assert_array_equal(a['baseline'], c['baseline'])
        np.testing.assert_array_equal(a['noise'], c['noise'])
        assert (a['signal'] != c['signal']).any()

    @pytest.mark.parametrize('model', synth.PEAK_MODELS)
    @pytest.mark.parametrize('plate', [False, True])
    def test_every_peak_is_inside_the_record(self, model, plate):
        # A peak cut by an end of the record is the endpoint mismatch
        # BEADS cannot repair.
        noise = 0.02
        shapes = dict(synth.DEFAULT_SHAPES, model=model)
        for s in range(8):
            rng = np.random.default_rng(s)
            n = int(rng.integers(500, 3000))
            d = synth.make_signal(n, 1, noise, rng, shapes=shapes,
                                  layout=_layout(plate_model=plate))
            for p in _analytes(d):
                y = _profile(d['x'], p)
                assert y[0] <= noise and y[-1] <= noise

    @pytest.mark.parametrize('model', synth.PEAK_MODELS)
    def test_nothing_elutes_before_the_dead_time(self, model):
        noise = 0.02
        shapes = dict(synth.DEFAULT_SHAPES, model=model)
        for s in range(8):
            d = synth.make_signal(2000, 1, noise, np.random.default_rng(s),
                                  shapes=shapes,
                                  layout=_layout(plate_model=True))
            t0 = d['meta']['dead_time']
            for p in _analytes(d):
                assert _profile(d['x'], p)[d['x'] <= t0].max() <= noise

    @pytest.mark.parametrize('plate', [False, True])
    def test_adjacent_analytes_are_resolved_at_p_0(self, plate):
        for s in range(8):
            rng = np.random.default_rng(s)
            n = int(rng.integers(500, 3000))
            d = synth.make_signal(n, 2, 0.02, rng,
                                  layout=_layout(plate_model=plate,
                                                 overlap_prob=0.))
            if d['meta']['n_analytes'] >= 2:
                assert d['meta']['rs_min'] >= synth.MIN_RESOLUTION - 1e-9

    def test_overlap_rights_follow_p(self):
        specs, _ = synth._draw_analytes(
            np.linspace(100., 2900., 2000), _layout(overlap_prob=0.2), 0.02,
            np.random.default_rng(3), _GAUSSIAN)
        assert np.mean([s['may_overlap'] for s in specs]) == pytest.approx(
            0.2, abs=0.03)

    def test_p_1_lets_peaks_overlap(self):
        rs = [synth.make_signal(500, 1, 0.02, np.random.default_rng(s),
                                layout=_layout(overlap_prob=1., n_peaks=30)
                                )['meta']['rs_min'] for s in range(10)]
        assert min(rs) < synth.MIN_RESOLUTION

    def test_p_moves_only_the_overlap_rights(self):
        # Every analyte takes its overlap variate whatever p, so at p = 0
        # and at a p no analyte uses, the signals are identical.
        a = synth.make_signal(3000, 1, 0.02, np.random.default_rng(6),
                              layout=_layout(overlap_prob=0.))
        b = synth.make_signal(3000, 1, 0.02, np.random.default_rng(6),
                              layout=_layout(overlap_prob=1e-12))
        np.testing.assert_array_equal(a['y'], b['y'])

    def test_fixed_widths_are_inside_their_range(self):
        lo, hi = synth.FIXED_SIGMA_RANGE
        for s in range(20):
            d = synth.make_signal(3000, 1, 0.02, np.random.default_rng(s))
            for p in _analytes(d):
                assert lo <= p['sigma'] <= hi

    def test_plate_widths_grow_with_the_place(self):
        d = synth.make_signal(6000, 1, 0.02, np.random.default_rng(2),
                              layout=_layout(plate_model=True, n_peaks=10))
        n_plates = d['meta']['n_plates']
        assert synth.PLATE_NUMBER_RANGE[0] <= n_plates
        assert n_plates <= synth.PLATE_NUMBER_RANGE[1]
        for p in _analytes(d):
            jitter = p['sigma'] * np.sqrt(n_plates) / p['tc']
            assert 0.85 - 1e-9 <= jitter <= 1.15 + 1e-9

    def test_width_modes_share_their_draws(self):
        # At one count the layouts take the same draws, so a peak kept in
        # both has one height, and the widths differ.
        a = synth.make_signal(30000, 1, 0.02, np.random.default_rng(1),
                              layout=_layout(n_peaks=20))
        b = synth.make_signal(30000, 1, 0.02, np.random.default_rng(1),
                              layout=_layout(n_peaks=20, plate_model=True))
        np.testing.assert_array_equal(a['noise'], b['noise'])
        np.testing.assert_array_equal(a['baseline'], b['baseline'])
        ha = {p['tc']: (p['height'], p['sigma']) for p in _analytes(a)}
        hb = {p['tc']: (p['height'], p['sigma']) for p in _analytes(b)}
        shared = ha.keys() & hb.keys()
        assert shared
        assert all(ha[c][0] == hb[c][0] for c in shared)
        assert any(ha[c][1] != hb[c][1] for c in shared)

    @pytest.mark.parametrize('model', synth.PEAK_MODELS)
    @pytest.mark.parametrize('layout', [
        dict(), dict(plate_model=True), dict(overlap_prob=1.),
        dict(plate_model=True, overlap_prob=0.5),
    ])
    def test_the_coverage_is_capped(self, model, layout):
        shapes = dict(synth.DEFAULT_SHAPES, model=model)
        top = []
        for s in range(10):
            d = synth.make_signal(500, 1, 0.02, np.random.default_rng(s),
                                  shapes=shapes,
                                  layout=_layout(n_peaks=30, **layout))
            m = d['meta']
            top.append(m['coverage'])
            assert m['coverage'] <= synth.MAX_COVERAGE
            assert m['n_requested'] == 30 == (
                m['n_analytes'] + m['dropped_early'] + m['dropped_cut']
                + m['dropped_overlap'] + m['dropped_coverage'])
            assert m['n_analytes'] == len(_analytes(d))
        assert max(top) > 0.8 * synth.MAX_COVERAGE

    def test_coverage_is_four_sigma_for_gaussians(self):
        t = np.arange(1000.)
        half = synth._FWHM_PER_SIGMA / 2.
        one = (500., 500. - 10. * half, 500. + 10. * half)
        assert synth._coverage(t, [one]) == pytest.approx(40. / 999.)
        # shared parts count once, parts outside the record not at all
        assert synth._coverage(t, [one, one]) == pytest.approx(40. / 999.)
        cut = (0., -10. * half, 10. * half)
        assert synth._coverage(t, [cut]) == pytest.approx(20. / 999.)
        assert synth._coverage(t, []) == 0.

    def test_peaks_keep_their_drawn_place(self):
        rng = np.random.default_rng(5)
        specs, _ = synth._draw_analytes(rng.uniform(0., 999., 30),
                                        _layout(), 0.02, rng)
        kept, _, _ = synth._keep_analytes(np.arange(1000.), specs, -np.inf,
                                          0.02)
        assert 0 < len(kept) < 30
        assert {p['tc'] for p in kept} <= {s['tc'] for s in specs}

    def test_the_cap_favours_no_part_of_the_record(self):
        # Dropping in the order of the draw leaves both halves as full.
        tc = np.concatenate([[p['tc'] for p in _analytes(synth.make_signal(
            500, 1, 0.02, np.random.default_rng(s), shapes=_GAUSSIAN,
            layout=_layout(n_peaks=30, overlap_prob=1.)))]
            for s in range(200)])
        share = np.mean(tc < 249.5)
        assert share == pytest.approx(0.5, abs=4. * 0.5 / np.sqrt(len(tc)))

    def test_a_signal_without_artifact_holds_a_peak(self):
        for s in range(100):
            d = synth.make_signal(500, 1, 0.02, np.random.default_rng(s),
                                  layout=_layout(n_peaks=1))
            assert d['meta']['n_analytes'] == 1

    def test_a_record_too_short_for_a_peak_is_refused(self):
        with pytest.raises(ValueError, match='no dead time or analyte'):
            synth.make_signal(5, 1, 0.02, np.random.default_rng(0),
                              layout=_layout(n_peaks=1))

    def test_true_apex_and_width_are_the_peaks_own(self):
        # `tc` and `fwhm_points` describe the Gaussian component of the
        # EMG; `apex` and `fwhm_true_points` must describe the peak.
        d = synth.make_signal(2000, 3, 0.02, np.random.default_rng(4),
                              layout=_layout(n_peaks=4))
        fine = np.linspace(d['x'][0], d['x'][-1], 40 * len(d['x']))
        for p in _analytes(d):
            y = _profile(fine, p)
            above = fine[y >= 0.5 * y.max()]
            assert p['apex'] == pytest.approx(fine[np.argmax(y)], abs=0.5)
            assert p['fwhm_true_points'] == pytest.approx(
                above[-1] - above[0], abs=1.)
            assert p['apex'] > p['tc']

    def test_record_lengths_stay_inside_the_range(self):
        rng = np.random.default_rng(0)
        lo, hi = synth.N_RANGE
        n = [synth.draw_record_length(rng) for _ in range(2000)]
        assert lo <= min(n) and max(n) <= hi


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
        d = synth.make_signal(1800, 1, 0.02, np.random.default_rng(7),
                              layout=_layout(n_peaks=8))
        analytes = _analytes(d)
        np.testing.assert_allclose(
            [p['tailing_factor'] for p in analytes],
            [1.09269172, 1.67352800, 1.40167753, 1.02971691,
             1.16233298, 1.26674689, 1.08002844], rtol=1e-7)
        np.testing.assert_allclose(
            [p['height'] for p in analytes],
            [8.5160365307, 1.2846652281, 6.4285163028, 1.4656452129,
             5.0982938279, 0.7715315198, 1.8928030057], rtol=1e-9)
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

    def test_heights_are_log_uniform_in_their_range(self):
        rng = np.random.default_rng(3)
        u = np.log([synth.draw_height(rng, 0.02, 60.) for _ in range(2000)])
        lo, hi = np.log(60. / synth.HEIGHT_SPAN), np.log(60.)
        assert lo <= u.min() and u.max() <= hi
        assert u.mean() == pytest.approx((lo + hi) / 2., abs=0.1)
        assert u.std() == pytest.approx((hi - lo) / np.sqrt(12.), abs=0.06)

    def test_no_height_is_below_the_snr_floor(self):
        rng = np.random.default_rng(5)
        h = [synth.draw_height(rng, 1., 60.) for _ in range(500)]
        assert min(h) >= synth.MIN_SNR

    @pytest.mark.parametrize('kind', synth.BASELINE_TYPES)
    def test_the_height_range_follows_the_baseline(self, kind):
        # Typically the tallest peak is 5 to 10 times the baseline.
        c, top = [], synth.HEIGHT_MAX[kind]
        for s in range(60):
            d = synth.make_signal(3000, kind, 0.02, np.random.default_rng(s),
                                  shapes=_GAUSSIAN)
            h = [p['height'] for p in _analytes(d)]
            assert max(h) <= top
            c.append(max(h) / d['meta']['baseline_range'])
            assert d['meta']['height_min_sigma'] == pytest.approx(
                min(h) / 0.02)
        assert 5. < np.median(c) < 10.

    def test_a_height_takes_one_draw(self):
        # One uniform per height, so the draws after it do not move.
        a, b = np.random.default_rng(13), np.random.default_rng(13)
        synth.draw_height(a, 0.05, 60.)
        b.uniform()
        assert a.uniform() == b.uniform()

    def test_models_share_everything_but_the_shapes(self):
        # Shapes and noise draw from streams of their own, so the same
        # seed gives the same dead time, noise and drawn peaks in every
        # model; which peaks are kept depends on their shape.
        made = {m: synth.make_signal(3000, 2, 0.05,
                                     np.random.default_rng(11),
                                     shapes=dict(synth.DEFAULT_SHAPES,
                                                 model=m),
                                     layout=_layout(plate_model=True))
                for m in synth.PEAK_MODELS}
        ref = made['emg']
        for m, d in made.items():
            np.testing.assert_array_equal(d['noise'], ref['noise'])
            np.testing.assert_array_equal(d['baseline'], ref['baseline'])
            assert d['meta']['dead_time'] == ref['meta']['dead_time']
            mine = {p['tc']: p['height'] for p in _analytes(d)}
            other = {p['tc']: p['height'] for p in _analytes(ref)}
            shared = mine.keys() & other.keys()
            assert shared
            assert all(mine[c] == other[c] for c in shared)
            assert {p['shape']['model'] for p in _analytes(d)} == {m}

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
        d = synth.make_signal(6000, 1, 0.02, np.random.default_rng(2),
                              shapes=shapes)
        analytes = _analytes(d)
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


class TestNamesAndManifest:
    def test_stem_puts_the_length_before_the_replicate(self):
        stem = synth.signal_stem('12a', 'emg', 2, 'snr20', 1874, 3)
        assert stem == 'SYN__12a__emg__type2__snr20__01874__0003'

    def test_peak_label_marks_the_artifact(self):
        assert synth.peak_label(0, True) == '0a'
        assert synth.peak_label(7, False) == '7'

    def test_manifest_row_matches_its_columns(self):
        d = synth.make_signal(900, 1, 0.02, np.random.default_rng(1),
                              layout=_layout(plate_model=True))
        stem = synth.signal_stem('x', 'emg', 1, 'snr25', 900, 0)
        row = synth.manifest_row(stem, 'snr25', 0, d).split(',')
        assert len(row) == len(synth.MANIFEST_COLUMNS)
        fields = dict(zip(synth.MANIFEST_COLUMNS, row))
        # the artifact lobes are not analytes
        assert int(fields['n_analytes']) == len(d['peaks']) - 2
        assert fields['plate_model'] == 'True'
        assert int(fields['baseline_type']) == 1


class TestProvenance:
    def _write(self, tmp_path, **overrides):
        params = dict(output='SET', seed=0, replicates=1, no_plot=False,
                      overlap_prob=0.)
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

    def test_records_the_exact_overlap_probability(self, tmp_path):
        line = self._write(tmp_path, overlap_prob=0.123456)[2]
        assert 'overlap_prob=0.123456' in line.split()

    def test_records_the_defaults(self, tmp_path):
        args = self._write(tmp_path)[2].split()[1:]
        assert 'seed=0' in args
        assert 'replicates=1' in args
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
    def _run(self, tmp_path, monkeypatch, *args):
        out = tmp_path / 'SET'
        monkeypatch.setattr(sys, 'argv', [
            'synth_dataset.py', str(out), '--no-plot', '--length', '600',
            *args])
        synth.main()
        return out

    def test_writes_provenance_after_the_manifest(self, tmp_path,
                                                  monkeypatch):
        out = self._run(tmp_path, monkeypatch)
        manifest = out / 'manifest.csv'
        provenance = out / 'PROVENANCE.txt'
        assert provenance.stat().st_mtime_ns >= manifest.stat().st_mtime_ns
        n_rows = len(manifest.read_text().splitlines()) - 1
        assert n_rows == len(list((out / 'signals').glob('*.txt'))) == 6
        assert 'replicates=1' in provenance.read_text()

    def test_the_seed_changes_the_set(self, tmp_path, monkeypatch):
        def signals(seed):
            out = self._run(tmp_path / seed, monkeypatch, '--seed', seed,
                            '--n-peaks', '2')
            return {f.stem.split('__')[3] + f.stem.split('__')[4]:
                    f.read_text() for f in (out / 'signals').glob('*.txt')}
        a, b, again = signals('0'), signals('1'), signals('00')
        assert a == again
        assert all(a[k] != b[k] for k in a)

    def test_each_noise_name_holds_its_own_ratio(self, tmp_path,
                                                 monkeypatch):
        out = self._run(tmp_path, monkeypatch, '--n-peaks', '2')
        for f in (out / 'truth').glob('*__truth.npz'):
            name = f.stem.split('__')[4]
            z = np.load(f)
            sigma = json.loads(str(z['meta']))['noise_sigma']
            ratio = 10. * np.log10(np.mean(z['baseline'] ** 2) / sigma ** 2)
            assert ratio == pytest.approx(synth.NOISE_CASES[name])

    def test_the_length_is_drawn_even_when_fixed(self, tmp_path,
                                                 monkeypatch):
        # Fixing the length at the one that was drawn changes nothing.
        base = ['synth_dataset.py', str(tmp_path / 'A'), '--no-plot',
                '--n-peaks', '2']
        monkeypatch.setattr(sys, 'argv', base)
        synth.main()
        f = sorted((tmp_path / 'A' / 'signals').glob('*type1*snr30*'))[0]
        n = int(f.stem.split('__')[5])
        base[1] = str(tmp_path / 'B')
        monkeypatch.setattr(sys, 'argv', base + ['--length', str(n)])
        synth.main()
        assert (tmp_path / 'B' / 'signals' / f.name).read_text() == \
            f.read_text()

    def test_the_legacy_types_come_on_request(self, tmp_path, monkeypatch):
        out = self._run(tmp_path, monkeypatch, '--legacy-baselines')
        kinds = {p.stem.split('__')[3] for p in (out / 'signals').glob('*')}
        assert kinds == {f'type{k}' for k in synth.BASELINE_TYPES}

    def test_flags_reach_the_set(self, tmp_path, monkeypatch):
        out = self._run(tmp_path, monkeypatch, '--n-peaks', '2',
                        '--plate-model',
                        '--overlap-prob', '0.5')
        names = sorted(p.stem for p in (out / 'signals').glob('*.txt'))
        for n in names:
            peaks = json.loads(str(np.load(
                out / 'truth' / f'{n}__truth.npz')['peaks']))
            left = sum(not p['artifact'] for p in peaks)
            assert n.split('__')[1] == f'{left}a'
        assert all(n.endswith('__00600__0000') for n in names)
        for f in (out / 'truth').glob('*__truth.npz'):
            meta = json.loads(str(np.load(f)['meta']))
            assert meta['plate_model']
            assert meta['overlap_prob'] == 0.5
            assert meta['n_points'] == 600
        first = next((out / 'signals').glob('*.txt')).read_text()
        assert first.split('\n')[1].split('\t')[0] == '1'

    @pytest.mark.parametrize('args', [
        ['--n-peaks', '0'],
        ['--n-peaks', '-1'],
        ['--overlap-prob', '1.5'],
        ['--length', '1'],
        ['--length', '499'],
    ])
    def test_the_parser_refuses_a_bad_layout(self, args, tmp_path,
                                             monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['synth_dataset.py',
                                          str(tmp_path / 'SET')] + args)
        with pytest.raises(SystemExit):
            synth.main()
        assert not (tmp_path / 'SET').exists()

    def test_refuses_a_non_empty_directory(self, tmp_path, monkeypatch):
        (tmp_path / 'old.txt').write_text('x')
        monkeypatch.setattr(sys, 'argv', ['synth_dataset.py', str(tmp_path)])
        with pytest.raises(SystemExit):
            synth.main()
        assert sorted(p.name for p in tmp_path.iterdir()) == ['old.txt']
