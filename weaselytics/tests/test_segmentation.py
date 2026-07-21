import numpy as np
import pytest

from weaselytics.segmentation import (
    classify_segments,
    pelt_linear,
    refine_candidates,
    segment_features,
    select_fcut,
    trim_candidates,
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
        assert not candidates[fcut_range < 0.5 / n_used].any()
        # ... but the region above the clip is not blanket-removed
        assert candidates.any()

    def test_frozen_tail_is_trimmed(self):
        fcut_range, y, segments = self._curve()
        candidates = trim_candidates(fcut_range, segments, 1000)
        # the last, noise-free stretch must not be a candidate
        assert not candidates[-100:].any()

    def test_shelf_remains_candidate(self):
        fcut_range, y, segments = self._curve()
        candidates = trim_candidates(fcut_range, segments, 1000)
        mid_shelf = 300 + 30 + 75
        assert candidates[mid_shelf]


class TestRefineCandidates:
    _N = 1000

    def _grid(self):
        return np.geomspace(1e-5, 0.5, self._N, endpoint=False)

    def _mask(self, *regions):
        mask = np.zeros(self._N, dtype=bool)
        for a, b in regions:
            mask[a:b] = True
        return mask

    def test_empty_mask_stays_empty(self):
        fcut_range = self._grid()
        refined = refine_candidates(fcut_range, self._mask())
        assert not refined.any()

    def test_sliver_region_is_removed(self):
        # 4.7 decades over 1000 points: 0.5 decades ~ 106 points
        fcut_range = self._grid()
        refined = refine_candidates(fcut_range,
                                    self._mask((100, 150), (300, 600)))
        assert not refined[100:150].any()
        assert refined[300:600].any()

    def test_all_slivers_keep_the_widest(self):
        fcut_range = self._grid()
        refined = refine_candidates(fcut_range,
                                    self._mask((100, 130), (300, 360)))
        assert refined[300:360].any()
        assert not refined[100:130].any()

    def test_third_region_is_removed(self):
        fcut_range = self._grid()
        refined = refine_candidates(
            fcut_range, self._mask((50, 250), (400, 600), (700, 900)))
        assert refined[400:600].any()
        assert not refined[700:900].any()

    def test_left_cut_of_first_region(self):
        fcut_range = self._grid()
        refined = refine_candidates(fcut_range, self._mask((200, 500)))
        # the geometric grid makes log-relative == index-relative
        assert not refined[200:230].any()
        assert refined[240:500].all()

    def test_right_cut_of_second_region(self):
        fcut_range = self._grid()
        refined = refine_candidates(fcut_range,
                                    self._mask((50, 250), (400, 700)))
        assert refined[400:560].all()
        assert not refined[575:700].any()
        # the second region is not left-cut (spanner optima sit at its
        # very beginning)
        assert refined[400]
