"""Tests for the synthetic-benchmark generator in ``tools/``.

``tools/`` is not a package, so the module is loaded by path. These
tests exist because the generator defines what "correct baseline" means
for every constant grounded against it: an error here does not produce a
wrong answer, it produces a wrong *target*.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

_PATH = (pathlib.Path(__file__).resolve().parents[2]
         / "tools" / "synth_dataset.py")
_spec = importlib.util.spec_from_file_location("synth_dataset", _PATH)
synth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synth)


class TestPearson7Peak:
    def test_height_at_centre(self):
        t = np.linspace(0., 10., 2001)
        y = synth.pearson7_peak(t, 5., 0.1, 10., 0.05, 3.0)
        assert y.max() == pytest.approx(3.0, rel=1e-9)
        assert t[np.argmax(y)] == pytest.approx(5.0, abs=5e-3)

    def test_symmetric_when_asymmetry_zero(self):
        # A_s = 0 must give an exactly even profile about the centre.
        t = np.linspace(-5., 5., 2001)
        y = synth.pearson7_peak(t, 0., 0.5, 8., 0.0, 1.0)
        np.testing.assert_allclose(y, y[::-1], rtol=0, atol=1e-12)

    def test_positive_asymmetry_tails_to_the_right(self):
        t = np.linspace(-5., 5., 4001)
        y = synth.pearson7_peak(t, 0., 0.3, 6., 0.25, 1.0)
        left = y[t < 0].sum()
        right = y[t > 0].sum()
        assert right > left

    def test_single_lobed_past_the_singularity(self):
        # The denominator vanishes at t - tc = -sigma / A_s; without the
        # clip the expression rises again into a spurious second lobe.
        sigma, a_s, tc = 0.0075, 0.28, 0.5
        t_sing = tc - sigma / a_s
        t = np.linspace(t_sing - 0.5, tc + 0.5, 20001)
        y = synth.pearson7_peak(t, tc, sigma, 10., a_s, 1.0)
        assert np.all(y[t < t_sing] == 0.0)
        assert np.all(np.isfinite(y))

    def test_larger_kurtosis_approaches_a_gaussian(self):
        # m -> infinity is the Gaussian limit; m small is Lorentzian,
        # which carries much heavier tails.
        t = np.linspace(-8., 8., 4001)
        sharp = synth.pearson7_peak(t, 0., 1., 500., 0.0, 1.0)
        heavy = synth.pearson7_peak(t, 0., 1., 1.5, 0.0, 1.0)
        far = np.abs(t) > 5
        assert heavy[far].sum() > 10 * sharp[far].sum()

    def test_rejects_bad_shape_parameters(self):
        t = np.linspace(0., 1., 11)
        with pytest.raises(ValueError, match="sigma"):
            synth.pearson7_peak(t, 0.5, 0.0, 10., 0.0, 1.0)
        with pytest.raises(ValueError, match="kurtosis"):
            synth.pearson7_peak(t, 0.5, 0.1, 0.0, 0.0, 1.0)

    def test_fitted_ranges_are_ordered(self):
        for name in ("PEARSON7_KURTOSIS_NIEZEN", "PEARSON7_KURTOSIS_LPYE",
                     "PEARSON7_KURTOSIS"):
            lo, hi = getattr(synth, name)
            assert 0 < lo < hi, name
        for name in ("PEARSON7_ASYMMETRY_NIEZEN", "PEARSON7_ASYMMETRY_LPYE",
                     "PEARSON7_ASYMMETRY"):
            lo, hi = getattr(synth, name)
            assert lo < hi, name

    def test_union_range_covers_both_provenances(self):
        # The generator must sample a chromatogram, not one instrument:
        # the union has to admit fronting peaks (LPYE) and the published
        # tailing range (Niezen).
        lo, hi = synth.PEARSON7_ASYMMETRY
        assert lo <= synth.PEARSON7_ASYMMETRY_LPYE[0]
        assert hi >= synth.PEARSON7_ASYMMETRY_NIEZEN[1]
        assert lo < 0, "fronting peaks must be representable"
        klo, khi = synth.PEARSON7_KURTOSIS
        assert klo <= synth.PEARSON7_KURTOSIS_LPYE[0]
        assert khi == synth.PEARSON7_KURTOSIS_NIEZEN[1]


class TestNoiseSigmaMad:
    def test_recovers_gaussian_sigma_on_flat_data(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0., 0.05, 20000)
        # Eq. (12b) works on consecutive differences, whose sd is
        # sigma*sqrt(2); the estimator is compared on that basis.
        est = synth.noise_sigma_mad(y, on_derivative=True) / np.sqrt(2)
        assert est == pytest.approx(0.05, rel=0.05)

    def test_derivative_form_is_robust_to_drift(self):
        # A large slow drift must not inflate Eq. (12b) the way it
        # inflates Eq. (12a) -- this is Niezen's stated reason for it.
        rng = np.random.default_rng(1)
        t = np.linspace(0., 1., 8000)
        y = rng.normal(0., 0.02, 8000) + 5.0 * t
        on_deriv = synth.noise_sigma_mad(y, on_derivative=True)
        on_signal = synth.noise_sigma_mad(y, on_derivative=False)
        assert on_signal > 10 * on_deriv


class TestPeakFreeStretch:
    def test_finds_the_quiet_half(self):
        # Peak width is realistic relative to the median filter, which
        # spans len(y)/PEAK_FREE_WINDOW_FRAC = 100 points here.
        rng = np.random.default_rng(2)
        n = 4000
        t = np.arange(n)
        y = rng.normal(0., 0.01, n)
        y += 5.0 * np.exp(-0.5 * ((t - 1000) / 15.) ** 2)
        r = synth.peak_free_stretch(y)
        assert r.stop - r.start > n // 3
        assert r.start > 1000        # the peak is excluded

    def test_broad_peak_can_be_absorbed(self):
        # KNOWN LIMITATION, pinned deliberately (synthetic_data.md §9).
        # A peak comparable to or wider than the median filter is
        # followed by it, so the residual stays small and the peak is
        # mistaken for drift. Any future change to the criterion should
        # either keep this behaviour or update this test knowingly.
        rng = np.random.default_rng(7)
        n = 4000
        t = np.arange(n)
        y = rng.normal(0., 0.01, n)
        y += 5.0 * np.exp(-0.5 * ((t - 1000) / 60.) ** 2)   # sigma >> N/40
        r = synth.peak_free_stretch(y)
        assert r.start < 1000 + 3 * 60, (
            "a broad peak is expected to slip through the criterion; "
            "if this now fails the detector improved -- update the doc")

    def test_whole_trace_when_featureless(self):
        rng = np.random.default_rng(3)
        y = rng.normal(0., 0.01, 3000)
        r = synth.peak_free_stretch(y)
        assert (r.stop - r.start) > 0.9 * len(y)

    def test_degenerate_input_returns_empty(self):
        assert synth.peak_free_stretch(np.zeros(500)) == slice(0, 0)
        assert synth.peak_free_stretch(np.array([1.0, 2.0])) == slice(0, 0)


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

    def test_reproduces_the_real_lattice_signature(self):
        # Quantised white noise at the real amplitude must show the same
        # signature as the instrument: consecutive differences that are
        # exact multiples of the step, and many exact ties.
        rng = np.random.default_rng(6)
        y = synth.quantise(rng.normal(0., 0.012, 20000))
        d = np.diff(y)
        ratio = d / synth.ADC_STEP_MV
        np.testing.assert_allclose(ratio, np.round(ratio), atol=1e-9)
        assert np.mean(d == 0) > 0.1

    def test_rejects_bad_step(self):
        with pytest.raises(ValueError, match="step"):
            synth.quantise(np.zeros(10), step=0.0)


class TestPybFamily:
    def test_gaussian_matches_pybaselines_bit_for_bit(self):
        # The transcribed formulas use _g in pybaselines' argument
        # order; it must agree with pybaselines.utils.gaussian EXACTLY,
        # otherwise every pyb signal silently differs from the source.
        from pybaselines.utils import gaussian
        x = np.linspace(0., 1000., 977)
        for h, c, s in ((6., 180., 5.), (20., 500., 500.),
                        (0.05, 400., 100.), (15., 400., 8.)):
            np.testing.assert_array_equal(synth._g(x, h, c, s),
                                          gaussian(x, h, c, s))

    def test_gaussian_agrees_with_gauss_peak_to_rounding(self):
        # Same function, different association of the division; they
        # agree to floating-point rounding but not bit-for-bit, which is
        # why _g does not delegate to gauss_peak.
        x = np.linspace(0., 1000., 977)
        np.testing.assert_allclose(synth._g(x, 6., 180., 5.),
                                   synth.gauss_peak(x, 180., 5., 6.),
                                   rtol=1e-12, atol=1e-300)

    def test_reproduces_the_documented_datasets(self):
        # Transcription check against the source: rebuild y1..y5 of
        # docs/algorithms/algorithms_1d/misc.rst here and compare.
        from pybaselines.utils import gaussian
        x = np.linspace(1, 1000, 500)
        signal = (gaussian(x, 6, 180, 5) + gaussian(x, 8, 350, 10)
                  + gaussian(x, 6, 550, 5) + gaussian(x, 9, 800, 10))
        signal_2 = (gaussian(x, 9, 100, 12) + gaussian(x, 15, 400, 8)
                    + gaussian(x, 13, 700, 12) + gaussian(x, 9, 880, 8))
        signal_3 = (gaussian(x, 8, 150, 10) + gaussian(x, 20, 120, 12)
                    + gaussian(x, 16, 300, 20) + gaussian(x, 12, 550, 5)
                    + gaussian(x, 20, 750, 12) + gaussian(x, 18, 800, 18)
                    + gaussian(x, 15, 830, 12))
        noise = np.random.default_rng(1).normal(0, 0.2, x.size)
        linear = 3 + 0.01 * x
        expected = {
            'B1_sparse_hi_noise': signal * 2 + linear + 5 * noise,
            'B2_dense': (signal + signal_2 + signal_3
                         + 5 + gaussian(x, 20, 500, 500) + noise),
            'B3_medium': signal + signal_2 + 5 + 15 * np.exp(-x / 400) + noise,
            'B4_lo_noise': (signal + signal_2 + 10 - 0.005 * x
                            + gaussian(x, 5, 850, 200) + noise * 0.5),
            'B5_negative_peaks': (signal * 2 - signal_2 + linear + 20
                                  + noise),
        }
        for case, y_expected in expected.items():
            got = synth.pyb_signal(case)
            np.testing.assert_allclose(got['y'], y_expected, rtol=0,
                                       atol=1e-12, err_msg=case)

    def test_endpoint_ladder_behaves_as_named(self):
        # The three [A] baselines exist to violate the BEADS periodicity
        # requirement to different degrees; the names must hold.
        def ends(case):
            b = synth.pyb_signal(case)['baseline']
            span = b.max() - b.min()
            return abs(b[0]) / span, abs(b[-1]) / span
        l0, r0 = ends('A0_ends_both')
        l1, r1 = ends('A1_ends_one')
        l2, r2 = ends('A2_ends_neither')
        assert l0 < 0.05 and r0 < 0.05          # zero at both ends
        assert min(l1, r1) < 0.05 < max(l1, r1)  # zero at exactly one
        assert min(l2, r2) > 0.05                # zero at neither

    def test_only_one_case_has_negative_peaks(self):
        neg = [c for c in synth.PYB_CASES
               if synth.pyb_signal(c)['meta']['has_negative_peaks']]
        assert neg == ['B5_negative_peaks']
        s = synth.pyb_signal('B5_negative_peaks')['signal']
        assert s.min() < -1.0, "the negative lobe must be substantial"

    def test_truth_is_exact(self):
        for case in synth.PYB_CASES:
            d = synth.pyb_signal(case)
            np.testing.assert_allclose(
                d['y'], d['signal'] + d['baseline'] + d['noise'],
                rtol=0, atol=1e-12, err_msg=case)

    def test_seed_override_changes_only_the_noise(self):
        a = synth.pyb_signal('B3_medium')
        b = synth.pyb_signal('B3_medium', seed=99)
        np.testing.assert_allclose(a['baseline'], b['baseline'])
        np.testing.assert_allclose(a['signal'], b['signal'])
        assert not np.allclose(a['noise'], b['noise'])
        assert b['meta']['seed'] == 99

    def test_rejects_unknown_names(self):
        with pytest.raises(ValueError, match="pyb case"):
            synth.pyb_signal('nope')
        with pytest.raises(ValueError, match="peak group"):
            synth.pyb_peaks(np.zeros(5), 'nope')
        with pytest.raises(ValueError, match="baseline kind"):
            synth.pyb_baseline(np.zeros(5), 'nope')


class TestPybRandomSignal:
    def test_is_a_pure_function_of_the_seed(self):
        a = synth.pyb_random_signal(11)
        b = synth.pyb_random_signal(11)
        for k in ('x', 'y', 'signal', 'baseline', 'noise'):
            np.testing.assert_array_equal(a[k], b[k])
        assert a['meta'] == b['meta']

    def test_different_seeds_differ(self):
        a = synth.pyb_random_signal(1)
        b = synth.pyb_random_signal(2)
        assert a['meta']['baseline_desc'] != b['meta']['baseline_desc'] \
            or a['meta']['n_peaks'] != b['meta']['n_peaks'] \
            or not np.allclose(a['noise'][:10], b['noise'][:10])

    def test_truth_is_exact(self):
        for seed in range(25):
            d = synth.pyb_random_signal(seed)
            np.testing.assert_allclose(
                d['y'], d['signal'] + d['baseline'] + d['noise'],
                rtol=0, atol=1e-12, err_msg=f'seed {seed}')

    def test_draws_stay_inside_the_declared_ranges(self):
        for seed in range(60):
            d = synth.pyb_random_signal(seed)
            m = d['meta']
            assert synth.PYB_N_POINTS[0] <= m['n_points'] <= synth.PYB_N_POINTS[1]
            assert synth.PYB_N_PEAKS[0] <= m['n_peaks'] <= synth.PYB_N_PEAKS[1]
            assert (synth.PYB_NOISE_STD_RANGE[0] <= m['noise_std']
                    <= synth.PYB_NOISE_STD_RANGE[1])
            for p in m['peaks']:
                assert (synth.PYB_PEAK_HEIGHT[0] * 0.999
                        <= abs(p['height'])
                        <= synth.PYB_PEAK_HEIGHT[1] * 1.001)
            assert 1 <= len(m['baseline_kinds']) <= 2
            assert set(m['baseline_kinds']) <= set(
                synth.PYB_BASELINE_COMPONENTS)

    def test_population_is_diverse(self):
        metas = [synth.pyb_random_signal(s)['meta'] for s in range(120)]
        # every baseline component must actually be reachable
        seen = set()
        for m in metas:
            seen.update(m['baseline_kinds'])
        assert seen == set(synth.PYB_BASELINE_COMPONENTS), seen
        # record lengths, peak counts and noise levels must all spread
        assert len({m['n_points'] for m in metas}) > 50
        assert len({m['n_peaks'] for m in metas}) >= 8
        stds = np.array([m['noise_std'] for m in metas])
        assert stds.max() / stds.min() > 10

    def test_negative_peaks_occur_but_are_the_minority(self):
        neg = [synth.pyb_random_signal(s)['meta']['has_negative_peaks']
               for s in range(200)]
        frac = np.mean(neg)
        assert 0.05 < frac < 0.45, frac
        # and when flagged, the signal really does dip negative
        for s in range(200):
            d = synth.pyb_random_signal(s)
            if d['meta']['has_negative_peaks']:
                assert d['signal'].min() < 0
                break

    def test_endpoint_condition_is_recorded_not_forced(self):
        # The periodicity axis must be measurable: some signals should
        # sit near zero at an end and others far from it.
        ends = np.array([synth.pyb_random_signal(s)['meta']['end_offsets']
                         for s in range(120)])
        assert ends.min() < 0.05, "no signal ends near its baseline minimum"
        assert ends.max() > 0.5, "no signal ends far from its minimum"

    def test_n_points_override(self):
        d = synth.pyb_random_signal(3, n_points=777)
        assert d['meta']['n_points'] == 777
        assert len(d['x']) == 777


_DONNIE = (pathlib.Path("/home/esteban/Simulation/DFT/separation_part2")
           / "donnie")


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
        # The transcribed expression is a Riemann sum missing its dx, so
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
    def test_components_sum_to_the_signal(self):
        d = synth.erb_signal('one_plateau')
        np.testing.assert_allclose(
            d['y'], d['signal'] + d['baseline'] + d['noise'], rtol=0,
            atol=1e-12)

    def test_default_seed_is_reproducible(self):
        a = synth.erb_signal('three_plateaus')
        b = synth.erb_signal('three_plateaus')
        np.testing.assert_array_equal(a['y'], b['y'])

    def test_explicit_seed_changes_only_the_noise(self):
        a = synth.erb_signal('one_plateau')
        b = synth.erb_signal('one_plateau', seed=12)
        np.testing.assert_array_equal(a['signal'], b['signal'])
        np.testing.assert_array_equal(a['baseline'], b['baseline'])
        assert not np.array_equal(a['noise'], b['noise'])

    def test_peaks_do_not_move_with_the_abscissa(self):
        # The three cases differ ONLY in where the fixed peak centres
        # fall inside the record; that is the whole mechanism of the
        # plateau-count knob.
        windows = {c: synth.erb_signal(c)['meta']['peak_window']
                   for c in synth.ERB_CASES}
        assert windows['one_plateau'][0] == pytest.approx(0.10, abs=1e-9)
        assert windows['two_plateaus'][1] == pytest.approx(0.22, abs=1e-9)
        assert windows['three_plateaus'][0] > 0.8

    def test_unknown_case_raises(self):
        with pytest.raises(ValueError):
            synth.erb_signal('four_plateaus')

    @pytest.mark.skipif(not _DONNIE.is_dir(),
                        reason="donnie/ reference files not present")
    @pytest.mark.parametrize("stem,case", [
        ("donnie1", "one_plateau"),
        ("donnie2", "two_plateaus"),
        ("donnie3", "three_plateaus"),
    ])
    def test_reproduces_the_exported_reference_signal(self, stem, case):
        # The strongest check available: the three exported signals in
        # donnie/ were written by running the source script, so an exact
        # match proves the transcription rather than merely testing it
        # against itself.
        ref = np.loadtxt(_DONNIE / f"{stem}.txt")[:, 1]
        got = synth.erb_signal(case, baseline_type=2)['y']
        assert np.abs(got - ref).max() < 1e-12


class TestErbRandomSignal:
    def test_pure_function_of_the_seed(self):
        a = synth.erb_random_signal(5)
        b = synth.erb_random_signal(5)
        np.testing.assert_array_equal(a['y'], b['y'])
        assert not np.array_equal(a['y'], synth.erb_random_signal(6)['y'])

    def test_components_sum_to_the_signal(self):
        d = synth.erb_random_signal(11)
        np.testing.assert_allclose(
            d['y'], d['signal'] + d['baseline'] + d['noise'], rtol=0,
            atol=1e-12)

    def test_peaks_land_inside_the_recorded_window(self):
        for seed in range(40):
            d = synth.erb_random_signal(seed)
            lo, hi = d['meta']['peak_window']
            span = d['x'][-1] - d['x'][0]
            frac = [(p['center'] - d['x'][0]) / span for p in d['meta']['peaks']]
            assert min(frac) >= lo - 1e-9, (seed, min(frac), lo)
            assert max(frac) <= hi + 1e-9, (seed, max(frac), hi)

    def test_window_never_runs_off_the_record(self):
        for seed in range(200):
            hi = synth.erb_random_signal(seed)['meta']['peak_window'][1]
            assert hi <= 0.98 + 1e-9, (seed, hi)

    def test_all_four_baselines_occur(self):
        kinds = {synth.erb_random_signal(s)['meta']['baseline_type']
                 for s in range(80)}
        assert kinds == {0, 1, 2, 3}, kinds

    def test_ranges_are_taken_from_the_source_not_widened(self):
        # Every drawn range must be spanned by values Erb's script
        # contains. Heights and sigmas come from his eight peaks, the
        # disabled one included; the window from his three cases.
        heights = [p[0] for p in synth.ERB_PEAKS] + [synth.ERB_PEAK_DISABLED[0]]
        assert synth.ERB_PEAK_HEIGHT == (min(heights), max(heights))
        sigmas = [p[2] for p in synth.ERB_PEAKS] + [synth.ERB_PEAK_DISABLED[2]]
        assert synth.ERB_PEAK_SIGMA_FRAC[0] == pytest.approx(min(sigmas) / 1000.)
        assert synth.ERB_PEAK_SIGMA_FRAC[1] == pytest.approx(max(sigmas) / 1000.)

    def test_n_points_and_baseline_overrides(self):
        d = synth.erb_random_signal(3, n_points=512, baseline_type=1)
        assert d['meta']['n_points'] == 512 and len(d['x']) == 512
        assert d['meta']['baseline_type'] == 1


class TestErbNativeSignal:
    def test_components_sum_to_the_signal_before_quantisation(self):
        rng = np.random.default_rng(0)
        d = synth.erb_native_signal(1800, 'multi_narrow', 2, 0.019, rng,
                                    quantise_output=False)
        np.testing.assert_allclose(
            d['y'], d['signal'] + d['baseline'] + d['noise'], rtol=0,
            atol=1e-12)

    def test_quantised_output_lands_on_the_detector_lattice(self):
        rng = np.random.default_rng(0)
        d = synth.erb_native_signal(900, 'multi_mixed', 1, 0.019, rng)
        ratio = d['y'] / synth.ADC_STEP_MV
        np.testing.assert_allclose(ratio, np.round(ratio), atol=1e-9)

    def test_reproducible_for_a_given_seed(self):
        a = synth.erb_native_signal(900, 'isocratic', 0, 0.019,
                                    np.random.default_rng(3))
        b = synth.erb_native_signal(900, 'isocratic', 0, 0.019,
                                    np.random.default_rng(3))
        np.testing.assert_array_equal(a['y'], b['y'])

    def test_peak_component_matches_the_native_family_exactly(self):
        # The whole point of the hybrid is that the peaks are native's,
        # not a second implementation of them. Same seed, same draws.
        n, case = 1800, 'multi_mixed'
        t1, sig1, pk1, t01 = synth.native_peak_component(
            n, case, np.random.default_rng(11))
        d = synth.erb_native_signal(n, case, 2, 0.019,
                                    np.random.default_rng(11))
        np.testing.assert_array_equal(d['signal'], sig1)
        np.testing.assert_array_equal(d['x'], t1)
        assert d['meta']['dead_time'] == t01
        assert len(d['peaks']) == len(pk1)

    def test_baseline_is_erbs_and_excludes_the_artefact(self):
        rng = np.random.default_rng(5)
        n = 1800
        d = synth.erb_native_signal(n, 'blank', 3, 0.019, rng)
        u = np.linspace(0., 1000., n)
        np.testing.assert_array_equal(
            d['baseline'], synth.erb_baseline(u, 3, exact_integral=True))
        # the injection artefact is in the signal, never in the truth
        art = [p for p in d['peaks'] if p.get('artifact')]
        assert art, "the injection artefact must be present"
        assert d['signal'].min() < 0, "its negative lobe must survive"

    def test_baseline_shape_does_not_depend_on_record_length(self):
        # erb_baseline is evaluated on a normalised abscissa, so the
        # four types stay comparable across the record lengths.
        rngs = [np.random.default_rng(1) for _ in range(2)]
        a = synth.erb_native_signal(900, 'blank', 2, 0.019, rngs[0])
        b = synth.erb_native_signal(4000, 'blank', 2, 0.019, rngs[1])
        assert (a['meta']['baseline_range']
                == pytest.approx(b['meta']['baseline_range'], rel=0.02))

    def test_all_four_baselines_and_every_peak_case_run(self):
        for case in synth.PEAK_CASES:
            for bt in range(4):
                d = synth.erb_native_signal(900, case, bt, 0.019,
                                            np.random.default_rng(0))
                assert np.all(np.isfinite(d['y']))
                assert d['meta']['n_analytes'] == synth.PEAK_CASES[case][0]

    def test_sampling_is_the_measured_rate(self):
        d = synth.erb_native_signal(1200, 'blank', 0, 0.019,
                                    np.random.default_rng(0))
        dt = np.median(np.diff(d['x']))
        assert dt == pytest.approx(1. / synth.PTS_PER_MIN, rel=1e-12)
