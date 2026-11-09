"""Tests for the cutoff overlay figure in ``tools/cutoff_overlay/``."""

import importlib.util
import pathlib

import numpy as np
import pytest

_PATH = (pathlib.Path(__file__).resolve().parents[2]
         / "tools" / "cutoff_overlay" / "cutoff_overlay.py")
_spec = importlib.util.spec_from_file_location("cutoff_overlay", _PATH)
overlay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(overlay)

_GRID = np.geomspace(0.00001, 0.5, num=1000, endpoint=False)


class TestStiffnessSpaced:
    def test_a_steady_change_gives_even_log_steps(self):
        def fit(f):
            return np.full(10, np.log10(f))

        got = overlay._stiffness_spaced(_GRID, fit, 5)
        steps = np.diff(np.log10(got))
        np.testing.assert_allclose(steps, steps[0], rtol=0.05)

    def test_one_jump_gives_no_repeats(self):
        def fit(f):
            return np.full(10, float(f > 1e-2))

        got = overlay._stiffness_spaced(_GRID, fit, 5)
        assert np.all(np.diff(got) > 0)
        assert len(got) < 5

    def test_no_fit_is_an_error(self):
        with pytest.raises(ValueError, match="no cutoff"):
            overlay._stiffness_spaced(_GRID, lambda f: None, 5)
