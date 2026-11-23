"""The training loop, two baselines, and the experiment runs."""

from __future__ import annotations

import csv
import json
import random
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torch import nn

from bikes_data import make_rentals
from model import BikeNet
from prepare import (
    SEED,
    RentalDataset,
    fit_preparer,
    make_loader,
    split_frame,
    target_values,
    transform_features,
)

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
EPOCHS = 80
MAIN_RUN = {
    "name": "main",
    "hidden": 64,
    "n_layers": 2,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": EPOCHS,
}
EXPERIMENTS = [
    MAIN_RUN,
    {
        "name": "high_learning_rate",
        "hidden": 64,
        "n_layers": 2,
        "learning_rate": 0.5,
        "batch_size": 64,
        "epochs": EPOCHS,
    },
    {
        "name": "wide_network",
        "hidden": 256,
        "n_layers": 4,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": EPOCHS,
    },
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, loss_fn, optimizer=None):
    """Train for one pass when an optimizer is given, otherwise only measure."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_absolute = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, target in loader:
            prediction = model(features)
            loss = loss_fn(prediction, target)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch_size = int(target.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_absolute += float(torch.abs(prediction.detach() - target).sum())
            count += batch_size
    return total_loss / count, total_absolute / count


def train_model(model, train_loader, val_loader, epochs: int, learning_rate: float):
    """Train for many epochs and restore the lowest validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        train_loss, _train_mae = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, loss_fn, optimizer=None)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history, best_epoch


def mean_absolute_error(actual, predicted) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def baseline_scores(x_train, y_train, x_val, y_val, x_test, y_test):
    """Training-mean and linear regression, scored in rentals."""
    train_mean = float(np.mean(y_train))
    mean_predictions = {
        "val_mae": mean_absolute_error(y_val, train_mean),
        "test_mae": mean_absolute_error(y_test, train_mean),
    }
    linear = LinearRegression()
    linear.fit(x_train, y_train)
    linear_predictions = {
        "val_mae": mean_absolute_error(y_val, linear.predict(x_val)),
        "test_mae": mean_absolute_error(y_test, linear.predict(x_test)),
    }
    return mean_predictions, linear_predictions


def network_mae(model, features, target) -> float:
    model.eval()
    with torch.no_grad():
        prediction = model(torch.as_tensor(features, dtype=torch.float32))
    return mean_absolute_error(target, prediction.numpy())


def save_curves(history, path: Path, title: str) -> None:
    epochs = [row["epoch"] for row in history]
    best = min(history, key=lambda row: row["val_loss"])
    figure, axis = plt.subplots()
    axis.plot(epochs, [row["train_loss"] for row in history], label="Training loss")
    axis.plot(epochs, [row["val_loss"] for row in history], label="Validation loss")
    axis.axvline(best["epoch"], color="gray", linestyle="--", label="Best epoch")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean squared error")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def run_network(config, x_train, y_train, x_val, y_val):
    set_seed(SEED)
    model = BikeNet(
        in_features=x_train.shape[1],
        hidden=config["hidden"],
        n_layers=config["n_layers"],
    )
    train_loader = make_loader(
        RentalDataset(x_train, y_train),
        batch_size=config["batch_size"],
        shuffle=True,
        seed=SEED,
    )
    val_loader = make_loader(
        RentalDataset(x_val, y_val),
        batch_size=config["batch_size"],
        shuffle=False,
        seed=SEED,
    )
    history, best_epoch = train_model(
        model,
        train_loader,
        val_loader,
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
    )
    return model, history, best_epoch


def write_history(path: Path, history) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["epoch", "train_loss", "val_loss", "val_mae"]
        )
        writer.writeheader()
        writer.writerows(history)


def write_experiments(path: Path, rows) -> None:
    fieldnames = [
        "run",
        "learning_rate",
        "hidden",
        "n_layers",
        "batch_size",
        "epochs",
        "best_epoch",
        "best_val_mae",
        "final_train_loss",
        "final_val_loss",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    set_seed(SEED)
    torch.set_num_threads(1)
    REPORTS.mkdir(exist_ok=True)
    frame = make_rentals()
    train_frame, val_frame, test_frame = split_frame(frame, seed=SEED)
    scaler, encoder = fit_preparer(train_frame)
    x_train = transform_features(train_frame, scaler, encoder)
    x_val = transform_features(val_frame, scaler, encoder)
    x_test = transform_features(test_frame, scaler, encoder)
    y_train = target_values(train_frame)
    y_val = target_values(val_frame)
    y_test = target_values(test_frame)

    trained = []
    experiment_rows = []
    for config in EXPERIMENTS:
        model, history, best_epoch = run_network(config, x_train, y_train, x_val, y_val)
        best_row = min(history, key=lambda row: row["val_loss"])
        trained.append((config, model, history, best_epoch, best_row))
        experiment_rows.append(
            {
                "run": config["name"],
                "learning_rate": config["learning_rate"],
                "hidden": config["hidden"],
                "n_layers": config["n_layers"],
                "batch_size": config["batch_size"],
                "epochs": config["epochs"],
                "best_epoch": best_epoch,
                "best_val_mae": best_row["val_mae"],
                "final_train_loss": history[-1]["train_loss"],
                "final_val_loss": history[-1]["val_loss"],
            }
        )
        save_curves(
            history,
            REPORTS / f"learning_curves_{config['name']}.png",
            config["name"],
        )
        print(
            f"{config['name']}: best epoch {best_epoch}, "
            f"val MAE {best_row['val_mae']:.3f}"
        )

    config, model, history, best_epoch, best_row = min(
        trained, key=lambda item: item[4]["val_loss"]
    )
    test_mae = network_mae(model, x_test, y_test)
    val_mae = network_mae(model, x_val, y_val)
    save_curves(history, REPORTS / "learning_curves.png", config["name"])
    mean_scores, linear_scores = baseline_scores(
        x_train, y_train, x_val, y_val, x_test, y_test
    )
    write_history(REPORTS / "history.csv", history)
    write_experiments(REPORTS / "experiments.csv", experiment_rows)
    metrics = {
        "seed": SEED,
        "chosen_run": config["name"],
        "best_epoch": best_epoch,
        "mean_baseline": mean_scores,
        "linear_baseline": linear_scores,
        "network": {
            "val_mae": val_mae,
            "test_mae": test_mae,
            "best_epoch": best_epoch,
            "best_val_loss": best_row["val_loss"],
            "hidden": config["hidden"],
            "n_layers": config["n_layers"],
            "learning_rate": config["learning_rate"],
        },
    }
    (REPORTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    by_hour = frame.groupby("hour")["rentals"].mean().round(2)
    print(by_hour.to_string())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
