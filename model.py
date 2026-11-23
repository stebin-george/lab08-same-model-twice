"""A small network that predicts a rental count."""

from torch import nn


class BikeNet(nn.Module):
    """At least two hidden ReLU layers, then one count with no activation."""

    def __init__(self, in_features: int, hidden: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError("n_layers must be at least 2.")
        layers: list[nn.Module] = []
        width = in_features
        for _ in range(n_layers):
            layers.append(nn.Linear(width, hidden))
            layers.append(nn.ReLU())
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features).squeeze(-1)
