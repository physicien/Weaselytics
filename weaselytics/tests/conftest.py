import os

import numpy as np
import pytest

os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg")


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SAMPLE_TXT = os.path.join(DATA_DIR, "sample_chromato.txt")


@pytest.fixture(scope="session")
def sample_txt_path():
    """Session-scoped: returns a constant path, no need to recompute."""
    return SAMPLE_TXT


@pytest.fixture(scope="session")
def sample_data():
    """Session-scoped: deterministic synthetic data reused across tests."""
    x = np.linspace(0, 20, 201)
    noise = 0.05 * np.random.default_rng(42).normal(size=len(x))
    y = 5.0 * np.exp(-0.5 * ((x - 10.0) / 1.5) ** 2) + noise
    return x, y


@pytest.fixture(scope="session")
def synthetic_gaussian():
    """Session-scoped: deterministic synthetic peak reused across tests."""
    x = np.linspace(0, 10, 101)
    y = 3.0 * np.exp(-0.5 * ((x - 5.0) / 0.8) ** 2)
    return x, y
