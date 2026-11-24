"""Train the main configuration in PyTorch and Keras on several seeds."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch

from model import BikeNet
from prepare import RentalDataset, make_loader
from train import MAIN_RUN, network_mae, set_seed, train_model
from train_keras import SEED, fit_keras, load_prepared, regression_mae

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
SEEDS = (SEED, SEED + 1, SEED + 2)
FIELDS = ["framework", "seed", "best_epoch", "val_mae", "test_mae", "seconds"]


def fit_torch(x_train, y_train, x_val, y_val, seed: int):
    """The Lab 07 loop, with this seed, on the main width and learning rate."""
    set_seed(seed)
    torch.set_num_threads(1)
    model = BikeNet(
        in_features=x_train.shape[1],
        hidden=MAIN_RUN["hidden"],
        n_layers=MAIN_RUN["n_layers"],
    )
    train_loader = make_loader(
        RentalDataset(x_train, y_train),
        batch_size=MAIN_RUN["batch_size"],
        shuffle=True,
        seed=seed,
    )
    val_loader = make_loader(
        RentalDataset(x_val, y_val),
        batch_size=MAIN_RUN["batch_size"],
        shuffle=False,
        seed=seed,
    )
    started = time.perf_counter()
    history, best_epoch = train_model(
        model,
        train_loader,
        val_loader,
        epochs=MAIN_RUN["epochs"],
        learning_rate=MAIN_RUN["learning_rate"],
    )
    seconds = time.perf_counter() - started
    return model, best_epoch, seconds, history


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    x_train, y_train, x_val, y_val, x_test, y_test = load_prepared()
    rows = []
    for seed in SEEDS:
        model, best_epoch, seconds, _history = fit_torch(
            x_train, y_train, x_val, y_val, seed
        )
        rows.append(
            {
                "framework": "pytorch",
                "seed": seed,
                "best_epoch": best_epoch,
                "val_mae": network_mae(model, x_val, y_val),
                "test_mae": network_mae(model, x_test, y_test),
                "seconds": seconds,
            }
        )
        print(
            f"pytorch seed {seed}: epoch {best_epoch}, "
            f"val MAE {rows[-1]['val_mae']:.3f}, "
            f"test MAE {rows[-1]['test_mae']:.3f}, {seconds:.1f}s"
        )
        keras_model, _rows, keras_epoch, keras_seconds = fit_keras(
            x_train, y_train, x_val, y_val, seed=seed
        )
        rows.append(
            {
                "framework": "keras",
                "seed": seed,
                "best_epoch": keras_epoch,
                "val_mae": regression_mae(keras_model, x_val, y_val),
                "test_mae": regression_mae(keras_model, x_test, y_test),
                "seconds": keras_seconds,
            }
        )
        print(
            f"keras seed {seed}: epoch {keras_epoch}, "
            f"val MAE {rows[-1]['val_mae']:.3f}, "
            f"test MAE {rows[-1]['test_mae']:.3f}, {keras_seconds:.1f}s"
        )
    path = REPORTS / "comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "val_mae": f"{row['val_mae']:.6f}",
                    "test_mae": f"{row['test_mae']:.6f}",
                    "seconds": f"{row['seconds']:.3f}",
                }
            )


if __name__ == "__main__":
    main()
