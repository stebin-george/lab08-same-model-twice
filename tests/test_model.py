import pytest
import torch

from model import BikeNet


@pytest.mark.parametrize("hidden", [8, 32, 128])
def test_network_returns_one_value_per_row(hidden):
    model = BikeNet(in_features=5, hidden=hidden, n_layers=2)
    output = model(torch.randn(7, 5))

    assert tuple(output.shape) == (7,)
