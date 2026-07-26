import numpy as np
import pytest

from weaselytics.segmentation import (
    classify_segments,
    detect_dips,
    dips_to_mask,
    instability_boundary,
    pelt_linear,
    segment_features,
    select_fcut,
    stability_dispersion,
    trim_candidates,
    trim_plateaus,
)


def synthetic_curve(noise_scale=1e-4, seed=0):
    """Piecewise curve mimicking an autocorrelation plot: a first
    plateau, a ramp down to a second plateau, a steep drop, then a noisy
    chaotic tail. Returns (fcut_range, r2, true_breakpoints)."""
    rng = np.random.default_rng(seed)
    fcut_range = np.geomspace(1e-5, 0.5, num=1000, endpoint=False)
    parts = [
        np.full(300, 0.99),
        np.linspace(0.99, 0.90, 100),
        np.full(250, 0.90),
        np.linspace(0.90, 0.10, 100),
        0.10 + 0.15 * np.abs(np.sin(np.arange(250) / 5.0)),
    ]
    r2 = np.concatenate(parts)
    r2 += rng.normal(0, noise_scale, len(r2))
    r2[750:] += rng.normal(0, 5e-2, 250)
    return fcut_range, r2, [300, 400, 650, 750, 1000]


class TestPeltLinear:
    def test_breakpoints_end_at_n(self):
        _, r2, _ = synthetic_curve()
        breakpoints = pelt_linear(r2)
        assert breakpoints[-1] == len(r2)

    def test_recovers_main_breakpoints(self):
        _, r2, true_bps = synthetic_curve()
        breakpoints = pelt_linear(r2)
        # Every true structural breakpoint has a detected breakpoint
        # within a small tolerance
        for true_bp in true_bps[:-1]:
            assert np.min(np.abs(breakpoints - true_bp)) <= 20

    def test_single_line_gives_one_segment(self):
        y = np.linspace(0.0, 1.0, 200)
        y += np.random.default_rng(3).normal(0, 1e-3, 200)
        breakpoints = pelt_linear(y)
        assert len(breakpoints) == 1

    def test_raises_on_short_data(self):
        with pytest.raises(ValueError):
            pelt_linear(np.zeros(10), min_size=15)


class TestSegmentFeatures:
    def test_flat_segment_has_small_relative_slope(self):
        fcut_range, r2, _ = synthetic_curve()
        breakpoints = pelt_linear(r2)
        segments = segment_features(fcut_range, r2, breakpoints)
        first = segments[0]
        assert first['rel_slope'] < 0.05
        assert first['mean'] == pytest.approx(0.99, abs=1e-2)

    def test_segments_are_contiguous(self):
        fcut_range, r2, _ = synthetic_curve()
        segments = segment_features(fcut_range, r2, pelt_linear(r2))
        for prev, seg in zip(segments, segments[1:]):
            assert seg['start'] == prev['end']
        assert segments[0]['start'] == 0
        assert segments[-1]['end'] == len(r2)


class TestClassifySegments:
    def test_plateaus_flat_and_drop_not(self):
        fcut_range, r2, _ = synthetic_curve()
        segments = segment_features(fcut_range, r2, pelt_linear(r2))
        segments = classify_segments(segments)
        # First segment (0.99 plateau) is flat
        assert segments[0]['flat']
        # The segment containing the steep drop (starting near index
        # 650) is not flat
        steep = [seg for seg in segments
                 if seg['start'] >= 620 and seg['end'] <= 780]
        assert steep
        assert not any(seg['flat'] for seg in steep)


class TestSelectFcut:
    def test_selects_edge_of_second_plateau(self):
        fcut_range, r2, _ = synthetic_curve()
        fcut, segments, chosen = select_fcut(fcut_range, r2)
        assert fcut is not None
        assert chosen is not None
        # The chosen plateau is the 0.90 one, right before the big drop
        assert segments[chosen]['mean'] == pytest.approx(0.90, abs=1e-2)
        # fcut is at the right edge of the chosen plateau
        end = segments[chosen]['end']
        assert fcut == pytest.approx(fcut_range[end - 1])
        # ... which lies near the true end of that plateau (index 650)
        assert 600 <= end <= 700

    def test_returns_none_when_nothing_is_flat(self):
        rng = np.random.default_rng(7)
        fcut_range = np.geomspace(1e-5, 0.5, num=200, endpoint=False)
        r2 = rng.normal(0.5, 0.2, 200)
        fcut, _, chosen = select_fcut(fcut_range, r2)
        assert fcut is None
        assert chosen is None


class TestClassifySegmentsHysteresis:
    def _staircase(self, with_final_cliff=True):
        """Staircase curve: p_ini, cliff, drifting shelf, then
        optionally a second cliff and a flat tail."""
        rng = np.random.default_rng(11)
        parts = [
            np.full(300, 1.0),
            np.linspace(1.0, 0.8, 30),
            np.linspace(0.8, 0.65, 150),
        ]
        if with_final_cliff:
            parts += [np.linspace(0.65, 0.3, 40), np.full(280, 0.3)]
        y = np.concatenate(parts) + rng.normal(0, 5e-4, sum(map(len, parts)))
        fcut_range = np.geomspace(1e-5, 0.5, len(y), endpoint=False)
        return fcut_range, y

    def _shelf_segment(self, fcut_range, y):
        segments = classify_segments(
            segment_features(fcut_range, y, pelt_linear(y)))
        mid = 300 + 30 + 75  # middle of the shelf
        return next(s for s in segments
                    if s['start'] <= mid < s['end'])

    def test_bracketed_shelf_is_flat(self):
        fcut_range, y = self._staircase(with_final_cliff=True)
        shelf = self._shelf_segment(fcut_range, y)
        # The shelf drifts too much for the tight tier but is accepted
        # by the loose, cliff-bracketed tier
        assert shelf['rel_slope'] > 0.2
        assert shelf['flat']

    def test_unbracketed_shelf_is_not_flat(self):
        fcut_range, y = self._staircase(with_final_cliff=False)
        shelf = self._shelf_segment(fcut_range, y)
        assert shelf['rel_slope'] > 0.2
        assert not shelf['flat']


class TestTrimCandidates:
    def _curve(self):
        """p_ini, cliff, shelf, cliff, live tail, then a frozen tail."""
        rng = np.random.default_rng(12)
        parts = [
            np.full(300, 1.0),
            np.linspace(1.0, 0.8, 30),
            np.linspace(0.8, 0.65, 150),
            np.linspace(0.65, 0.3, 40),
            np.full(180, 0.3),
        ]
        y = np.concatenate(parts) + rng.normal(0, 5e-4, sum(map(len, parts)))
        y = np.concatenate([y, np.full(300, 0.3)])  # noise-free frozen tail
        fcut_range = np.geomspace(1e-5, 0.5, len(y), endpoint=False)
        segments = classify_segments(
            segment_features(fcut_range, y, pelt_linear(y)))
        return fcut_range, y, segments

    def test_sub_fundamental_region_is_trimmed(self):
        fcut_range, y, segments = self._curve()
        n_used = 1000
        candidates = trim_candidates(fcut_range, segments, n_used)
        assert not candidates[fcut_range < 1.0 / n_used].any()
        # ... but the region above the clip is not blanket-removed
        assert candidates.any()

    def test_frozen_tail_is_trimmed(self):
        fcut_range, y, segments = self._curve()
        candidates = trim_candidates(fcut_range, segments, 1000)
        # the last, noise-free stretch must not be a candidate
        assert not candidates[-100:].any()

    # The shelf of `_curve` sits at fcut ~8e-4, so with `n_used=1000`
    # the sub-fundamental clip (1/n_used = 1e-3 at the default c1=1.0)
    # would swallow it. These two tests are about the collapse
    # exclusion and about the shelf surviving, not about the clip, so
    # they use a longer record whose fundamental falls below the shelf.
    N_USED_LONG = 3000

    def test_collapse_exclusion_removes_low_plateau(self):
        # The curve has a high shelf (r2 ~ 0.72) and a low live-tail
        # plateau (r2 ~ 0.3) past the drop. Without the gate both are
        # candidates; with it, the low plateau is removed as past the
        # collapse while the high shelf survives.
        fcut_range, y, segments = self._curve()
        base = trim_candidates(fcut_range, segments, self.N_USED_LONG)
        assert base[600]                      # low live tail, kept
        excl = trim_candidates(fcut_range, segments, self.N_USED_LONG,
                               exclude_collapse=True)
        assert not excl[600]                  # low live tail, dropped
        assert excl[400]                      # high shelf, survives

    def test_shelf_remains_candidate(self):
        fcut_range, y, segments = self._curve()
        candidates = trim_candidates(fcut_range, segments,
                                     self.N_USED_LONG)
        mid_shelf = 300 + 30 + 75
        assert candidates[mid_shelf]


def staircase_curve(shelf=True, noise_scale=1e-4, seed=0):
    """Autocorrelation-like curve with (or without) a mid-descent shelf.

    Top plateau, a first drop, an optional gentle shelf (the
    proto-plateau), a steep drop to the global minimum, then a rising
    tail. Returns (fcut_range, r2, shelf_slice)."""
    rng = np.random.default_rng(seed)
    fcut_range = np.geomspace(1e-5, 0.5, num=1000, endpoint=False)
    if shelf:
        parts = [
            np.full(300, 1.00),
            np.linspace(1.00, 0.85, 80),
            np.linspace(0.85, 0.80, 150),      # shelf: proto-plateau
            np.linspace(0.80, 0.05, 120),
            np.linspace(0.05, 0.30, 350),      # rising tail
        ]
        shelf_slice = slice(380, 530)
    else:
        parts = [
            np.full(300, 1.00),
            np.linspace(1.00, 0.05, 350),      # single steep drop
            np.linspace(0.05, 0.30, 350),
        ]
        shelf_slice = slice(0, 0)
    r2 = np.concatenate(parts)
    r2 += rng.normal(0, noise_scale, len(r2))
    return fcut_range, r2, shelf_slice


class TestDetectDips:
    def test_finds_the_proto_plateau(self):
        fcut_range, r2, shelf = staircase_curve(shelf=True)
        dips = detect_dips(fcut_range, r2)
        floors = [d['floor'] for d in dips]
        assert any(shelf.start <= f < shelf.stop for f in floors)

    def test_single_step_curve_has_no_dip(self):
        fcut_range, r2, _ = staircase_curve(shelf=False)
        assert detect_dips(fcut_range, r2) == []

    def test_prominence_floor_rejects_the_shelf(self):
        fcut_range, r2, _ = staircase_curve(shelf=True)
        assert detect_dips(fcut_range, r2, min_prominence=0.9) == []

    def test_level_guards_drop_the_collapse_floor(self):
        # The rising tail turns at the global minimum, so a dip there
        # would sit at level ~0; every accepted dip must clear level_min.
        fcut_range, r2, _ = staircase_curve(shelf=True)
        dips = detect_dips(fcut_range, r2)
        assert all(d['level'] > 0.08 for d in dips)
        assert all(d['level'] < 0.92 for d in dips)

    def test_dips_are_before_the_global_minimum(self):
        fcut_range, r2, _ = staircase_curve(shelf=True)
        imin = int(np.argmin(r2))
        assert all(d['floor'] < imin for d in detect_dips(fcut_range, r2))

    def test_flat_curve_returns_empty(self):
        fcut_range = np.geomspace(1e-5, 0.5, num=1000, endpoint=False)
        r2 = np.full(1000, 0.5)
        assert detect_dips(fcut_range, r2) == []

    def test_mask_covers_the_basins(self):
        fcut_range, r2, _ = staircase_curve(shelf=True)
        dips = detect_dips(fcut_range, r2)
        mask = dips_to_mask(fcut_range, dips)
        assert mask.dtype == bool and len(mask) == len(fcut_range)
        for dip in dips:
            assert mask[dip['floor']]
            assert mask[dip['start']:dip['end'] + 1].all()

    def test_empty_dips_give_empty_mask(self):
        fcut_range, r2, _ = staircase_curve(shelf=False)
        mask = dips_to_mask(fcut_range, detect_dips(fcut_range, r2))
        assert not mask.any()


class TestTrimPlateaus:
    def _curve(self):
        """p_ini, cliff, shelf, cliff, low live tail, frozen tail — with
        a proto-plateau shelf that detect_dips picks up."""
        rng = np.random.default_rng(12)
        parts = [
            np.full(300, 1.0),
            np.linspace(1.0, 0.8, 30),
            np.linspace(0.8, 0.65, 150),
            np.linspace(0.65, 0.3, 40),
            np.full(180, 0.3),
        ]
        y = np.concatenate(parts) + rng.normal(0, 5e-4, sum(map(len, parts)))
        y = np.concatenate([y, np.full(300, 0.3)])   # frozen tail
        fcut_range = np.geomspace(1e-5, 0.5, len(y), endpoint=False)
        segments = classify_segments(
            segment_features(fcut_range, y, pelt_linear(y)))
        dips = detect_dips(fcut_range, y)
        return fcut_range, y, segments, dips

    def test_no_snr_removed_without_collapse(self):
        fcut_range, y, segments, dips = self._curve()
        masks = trim_plateaus(fcut_range, segments, dips, 1000,
                              exclude_collapse=False)
        assert not masks['snr_removed'].any()

    def test_surviving_disjoint_from_removed(self):
        fcut_range, y, segments, dips = self._curve()
        for excl in (False, True):
            masks = trim_plateaus(fcut_range, segments, dips, 1000,
                                  exclude_collapse=excl)
            assert not (masks['surviving'] & masks['removed']).any()
            assert not (masks['surviving'] & masks['snr_removed']).any()

    def test_sub_fundamental_is_removed(self):
        fcut_range, y, segments, dips = self._curve()
        n_used = 1000
        masks = trim_plateaus(fcut_range, segments, dips, n_used)
        sub = fcut_range < 1.0 / n_used
        # nothing below the fundamental survives ...
        assert not masks['surviving'][sub].any()

    def test_collapse_removes_the_low_tail(self):
        # The low live tail (r2 ~ 0.3) is past the collapse; with the
        # gate on it must move into snr_removed and out of surviving.
        fcut_range, y, segments, dips = self._curve()
        off = trim_plateaus(fcut_range, segments, dips, 1000,
                            exclude_collapse=False)
        on = trim_plateaus(fcut_range, segments, dips, 1000,
                           exclude_collapse=True)
        assert off['surviving'][600] and not on['surviving'][600]
        assert on['snr_removed'][600]


class TestInstabilityBoundary:
    """The stiff-side exclusion driven by the baseline-stability curve.

    Its thresholds are not grounded (see `instability_boundary`), so
    these tests pin the BEHAVIOUR of the rule -- fires only when the
    fundamental sits in a flailing region, and stops once the
    oscillations settle -- not the particular values.
    """

    def _curve(self, flail_lo, flail_hi, amp=1.0, n=1000):
        """Stability curve that flails between two cutoffs and is quiet
        elsewhere."""
        rng = np.random.default_rng(3)
        fcut_range = np.geomspace(1e-5, 0.5, num=n, endpoint=False)
        stability = np.full(n, 1e-3)
        band = (fcut_range >= flail_lo) & (fcut_range <= flail_hi)
        stability[band] = rng.uniform(0, amp, band.sum())
        return fcut_range, stability

    def test_fires_when_the_fundamental_is_inside_the_flailing(self):
        n_used = 1000                      # fundamental at 1e-3
        fcut_range, stability = self._curve(2e-4, 5e-3)
        boundary = instability_boundary(fcut_range, stability, n_used)
        assert boundary is not None
        # the exclusion reaches past the fundamental, and stops inside
        # the quiet zone beyond the flailing
        assert boundary > 1.0 / n_used
        assert boundary >= 5e-3

    def test_silent_when_the_fundamental_is_in_a_quiet_zone(self):
        # same flailing band, but a much shorter record puts the
        # fundamental above it, where the curve is already settled
        fcut_range, stability = self._curve(1e-4, 5e-4)
        assert instability_boundary(fcut_range, stability, 100) is None

    def test_silent_on_the_flexible_ramp(self):
        # The flexible side: stability climbs smoothly toward the
        # collapse instead of scattering. A ramp of realistic height
        # (the reference signals reach ~0.16) leaves the fundamental
        # quiet, so no stiff-side exclusion is triggered.
        fcut_range = np.geomspace(1e-5, 0.5, num=1000, endpoint=False)
        stability = np.linspace(0, 0.16, 1000)
        assert instability_boundary(fcut_range, stability, 1000) is None

    def test_dispersion_is_far_smaller_on_a_ramp_than_on_scatter(self):
        # Note what this does NOT claim: the dispersion of a window is
        # the spread of the values in it, so a steep enough ramp does
        # register (0 -> 50 over the grid gives ~1.05 at fcut 1e-3, well
        # above the trigger). What separates the two sides in practice
        # is that the flexible ramp is gentle AND far from the
        # fundamental, where the rule is evaluated.
        fcut_range = np.geomspace(1e-5, 0.5, num=1000, endpoint=False)
        rng = np.random.default_rng(0)
        amplitude = 1.0
        d_ramp = stability_dispersion(
            fcut_range, np.linspace(0, amplitude, 1000))
        d_scatter = stability_dispersion(
            fcut_range, rng.uniform(0, amplitude, 1000))
        # measured separation is a factor of ~4.3 between the ramp's
        # worst window and the scatter's quietest one
        assert d_ramp.max() < 0.5 * d_scatter.min()

    def test_trim_plateaus_reports_the_extra_cut(self):
        fcut_range, r2, _ = synthetic_curve()
        segments = classify_segments(
            segment_features(fcut_range, r2, pelt_linear(r2)))
        dips = detect_dips(fcut_range, r2)
        _, stability = self._curve(2e-4, 5e-3)
        off = trim_plateaus(fcut_range, segments, dips, 1000)
        on = trim_plateaus(fcut_range, segments, dips, 1000,
                           stability=stability)
        assert not off['instab_removed'].any()
        assert on['instab_removed'].any()
        # what it removes came out of the survivors, and nothing else
        assert not (on['surviving'] & on['instab_removed']).any()
        assert (on['surviving'] | on['instab_removed']).sum() == \
            off['surviving'].sum()
