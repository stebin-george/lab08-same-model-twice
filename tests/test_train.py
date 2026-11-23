from copy import deepcopy

import numpy as np
import torch
from torch import nn

from model import BikeNet
from prepare import RentalDataset, make_loader
from train import run_epoch, train_model


def _loader(rows=8, features=3, batch_size=4, seed=0):
    torch.manual_seed(seed)
    values = torch.randn(rows, features).numpy()
    target = values[:, 0] * 1.5
    dataset = RentalDataset(values, np.asarray(target, dtype=np.float32))
    return make_loader(dataset, batch_size=batch_size, shuffle=False, seed=seed)


def test_one_optimizer_step_changes_every_layer():
    torch.manual_seed(0)
    model = BikeNet(in_features=3, hidden=8, n_layers=2)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    run_epoch(model, _loader(), nn.MSELoss(), optimizer)

    for name, value in model.named_parameters():
        assert not torch.equal(before[name], value)


def test_each_batch_starts_with_cleared_gradients():
    torch.manual_seed(0)
    model = BikeNet(in_features=2, hidden=4, n_layers=2)
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-0.5, 0.5]], dtype=torch.float32
    )
    target = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.float32)
    loader = make_loader(
        RentalDataset(features.numpy(), target.numpy()),
        batch_size=2,
        shuffle=False,
        seed=0,
    )
    reference = deepcopy(model)
    loss = nn.MSELoss()(reference(features[2:]), target[2:])
    loss.backward()
    expected = [value.grad.detach().clone() for value in reference.parameters()]

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    run_epoch(model, loader, nn.MSELoss(), optimizer)

    for actual, wanted in zip(model.parameters(), expected, strict=True):
        assert torch.allclose(actual.grad, wanted)


def test_one_batch_can_be_memorized():
    torch.manual_seed(0)
    features = torch.randn(12, 3)
    target = features[:, 0] * 2 + 0.3
    loader = make_loader(
        RentalDataset(features.numpy(), target.numpy()),
        batch_size=12,
        shuffle=False,
        seed=0,
    )
    model = BikeNet(in_features=3, hidden=32, n_layers=2)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    start, _mae = run_epoch(model, loader, loss_fn, optimizer=None)
    for _step in range(400):
        run_epoch(model, loader, loss_fn, optimizer)
    finish, _mae = run_epoch(model, loader, loss_fn, optimizer=None)

    assert finish < start * 0.1


def test_validation_epoch_leaves_weights_unchanged():
    torch.manual_seed(0)
    model = BikeNet(in_features=3, hidden=8, n_layers=2)
    before = deepcopy(model.state_dict())

    run_epoch(model, _loader(), nn.MSELoss(), optimizer=None)

    for name, value in model.state_dict().items():
        assert torch.equal(before[name], value)


def test_training_logs_every_epoch_and_restores_the_best_weights(monkeypatch):
    model = BikeNet(in_features=3, hidden=4, n_layers=2)
    answers = [
        (1.0, 3.0),
        (2.0, 4.0),
        (0.5, 1.0),
        (0.2, 0.4),
        (0.1, 0.2),
        (5.0, 8.0),
    ]
    snapshots = []
    calls = {"count": 0}

    def fake_epoch(model, loader, loss_fn, optimizer=None):
        if optimizer is not None:
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(1.0)
            snapshots.append(deepcopy(model.state_dict()))
        answer = answers[calls["count"]]
        calls["count"] += 1
        return answer

    monkeypatch.setattr("train.run_epoch", fake_epoch)
    history, best_epoch = train_model(model, None, None, epochs=3, learning_rate=0.01)

    assert best_epoch == 2
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert set(history[0]) == {"epoch", "train_loss", "val_loss", "val_mae"}
    for name, value in model.state_dict().items():
        assert torch.equal(value, snapshots[1][name])
