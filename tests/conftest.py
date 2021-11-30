import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dummy_df():
    rows = [
        {"text": "testing row 0", "target": 0},
        {"text": "testing row 1", "target": 0},
        {"text": "testing row 2", "target": 1},
        {"text": "testing row 3", "target": 1},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def dummy_clusters():
    return {"0": [0, 1], "1": [2, 3]}


@pytest.fixture
def dummy_embeddings():
    return (
        np.array([[0.1, 0.2], [-0.1, -0.1], [0.5, -0.8], [0.4, -0.9]]),
        np.array([0.1, -0.1, 0.5, 0.4]),
        np.array([0.2, -0.1, -0.8, -0.9]),
    )
