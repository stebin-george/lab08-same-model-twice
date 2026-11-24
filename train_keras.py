"""Train the main Lab 07 configuration with Keras compile and fit."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import keras
import numpy as np

from bikes_data import make_rentals
from keras_model import build_bike_model
from prepare import (
    SEED,
    fit_preparer,
    split_frame,
    target_values,
    transform_features,
)
from train import MAIN_RUN, save_curves

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
CHECKPOINT = REPORTS / "keras_main.keras"


def load_prepared(seed: int = SEED):
    """The same split and the same fitted scaler as the PyTorch script."""
    frame = make_rentals()
    train_frame, val_frame, test_frame = split_frame(frame, seed=seed)
    scaler, encoder = fit_preparer(train_frame)
    return (
        transform_features(train_frame, scaler, encoder),
        target_values(train_frame),
        transform_features(val_frame, scaler, encoder),
        target_values(val_frame),
        transform_features(test_frame, scaler, encoder),
        target_values(test_frame),
    )


def _adam(learning_rate: float):
    """Match PyTorch Adam: betas 0.9 and 0.999, epsilon 1e-8."""
    return keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
    )


def history_rows(history) -> list[dict]:
    """One row per epoch, numbered from 1, same columns as the PyTorch log."""
    losses = history.history["loss"]
    val_losses = history.history["val_loss"]
    val_maes = history.history["val_mae"]
    rows = []
    for epoch, train_loss, val_loss, val_mae in zip(
        range(1, len(losses) + 1), losses, val_losses, val_maes, strict=True
    ):
        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_mae": float(val_mae),
            }
        )
    return rows


def fit_keras(
    x_train,
    y_train,
    x_val,
    y_val,
    seed: int,
    epochs: int = MAIN_RUN["epochs"],
    hidden: int = MAIN_RUN["hidden"],
    n_layers: int = MAIN_RUN["n_layers"],
    learning_rate: float = MAIN_RUN["learning_rate"],
    batch_size: int = MAIN_RUN["batch_size"],
    checkpoint_path: Path | None = None,
):
    """Train for every epoch, then put back the lowest validation loss.

    Patience equals the number of epochs, so EarlyStopping never cuts the
    run short. Both frameworks therefore train for the same length.
    """
    keras.utils.set_random_seed(seed)
    model = build_bike_model(x_train.shape[1], hidden=hidden, n_layers=n_layers)
    model.compile(
        optimizer=_adam(learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=epochs,
            restore_best_weights=True,
        )
    ]
    if checkpoint_path is not None:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
            )
        )
    started = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
        callbacks=callbacks,
    )
    seconds = time.perf_counter() - started
    rows = history_rows(history)
    best_epoch = min(rows, key=lambda row: row["val_loss"])["epoch"]
    return model, rows, best_epoch, seconds


def regression_mae(model, features, target) -> float:
    """Flatten (N, 1) predictions before comparing them with shape (N,)."""
    prediction = np.asarray(model.predict(features, verbose=0)).reshape(-1)
    return float(np.mean(np.abs(target - prediction)))


def write_history(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["epoch", "train_loss", "val_loss", "val_mae"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    x_train, y_train, x_val, y_val, x_test, y_test = load_prepared()
    model, rows, best_epoch, seconds = fit_keras(
        x_train,
        y_train,
        x_val,
        y_val,
        seed=SEED,
        checkpoint_path=CHECKPOINT,
    )
    val_mae = regression_mae(model, x_val, y_val)
    test_mae = regression_mae(model, x_test, y_test)
    write_history(REPORTS / "history_keras.csv", rows)
    save_curves(rows, REPORTS / "learning_curves_keras.png", "keras main")
    print(
        f"keras main: best epoch {best_epoch}, "
        f"val MAE {val_mae:.3f}, test MAE {test_mae:.3f}, "
        f"{seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
