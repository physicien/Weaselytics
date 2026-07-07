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
    return SAMPLE_TXT


@pytest.fixture(scope="session")
def sample_data():
    x = np.linspace(0, 20, 201)
    noise = 0.05 * np.random.default_rng(42).normal(size=len(x))
    y = 5.0 * np.exp(-0.5 * ((x - 10.0) / 1.5) ** 2) + noise
    return x, y


@pytest.fixture(scope="session")
def synthetic_gaussian():
    x = np.linspace(0, 10, 101)
    y = 3.0 * np.exp(-0.5 * ((x - 5.0) / 0.8) ** 2)
    return x, y
