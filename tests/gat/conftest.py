import pytest
from models.gat.layers import ConfGATLayer


@pytest.fixture(scope="module")
def N():
    return 10

@pytest.fixture(scope="module")
def c_gat_layer():
    return ConfGATLayer(
        in_features=10,
        out_features=10,
        n_head=2,
        dropout_rate=0.1,
        alpha=0.2,
        concat=True,
        shared_weight=True,
    )