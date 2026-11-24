"""Checks for the Keras network, the weight copy, and a short training run."""

import keras
import numpy as np
import pytest
import torch

from keras_model import build_bike_model, copy_torch_weights
from model import BikeNet
from prepare import (
    RentalDataset,
    fit_preparer,
    split_frame,
    target_values,
    transform_features,
)
from train_keras import fit_keras, regression_mae


def test_keras_returns_one_value_per_row():
    model = build_bike_model(in_features=6, hidden=8, n_layers=2)
    features = np.zeros((7, 6), dtype=np.float32)
    prediction = np.asarray(model.predict(features, verbose=0))
    assert prediction.shape == (7, 1)


@pytest.mark.parametrize("hidden", [8, 32, 128])
def test_parameter_count_matches_pytorch(hidden):
    torch_model = BikeNet(in_features=6, hidden=hidden, n_layers=2)
    keras_model = build_bike_model(in_features=6, hidden=hidden, n_layers=2)
    torch_count = sum(parameter.numel() for parameter in torch_model.parameters())
    assert torch_count == keras_model.count_params()


def test_copied_weights_predict_the_same_values():
    torch.manual_seed(0)
    torch_model = BikeNet(in_features=4, hidden=5, n_layers=2)
    with torch.no_grad():
        for parameter in torch_model.parameters():
            values = torch.arange(parameter.numel(), dtype=torch.float32)
            parameter.copy_(values.reshape(parameter.shape) * 0.01)
    keras_model = build_bike_model(in_features=4, hidden=5, n_layers=2)
    copy_torch_weights(torch_model, keras_model)
    features = np.array(
        [[0.2, -1.0, 0.5, 3.0], [1.0, 0.0, -0.3, 0.7]], dtype=np.float32
    )
    with torch.no_grad():
        torch_prediction = torch_model(torch.as_tensor(features)).numpy()
    keras_prediction = np.asarray(keras_model.predict(features, verbose=0)).reshape(-1)
    np.testing.assert_allclose(torch_prediction, keras_prediction, atol=1e-5)


def test_keras_and_pytorch_receive_the_same_rows(rentals):
    train_frame, val_frame, _test_frame = split_frame(rentals)
    scaler, encoder = fit_preparer(train_frame)
    features = transform_features(val_frame, scaler, encoder)
    target = target_values(val_frame)
    dataset = RentalDataset(features, target)
    assert np.array_equal(dataset.features.numpy(), features)
    assert np.array_equal(dataset.target.numpy(), target)


def test_keras_can_memorize_one_batch():
    features = np.random.default_rng(0).normal(size=(12, 3)).astype(np.float32)
    target = features[:, 0] * 2 + 0.3
    keras.utils.set_random_seed(0)
    model = build_bike_model(in_features=3, hidden=16, n_layers=2)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.05, epsilon=1e-8),
        loss="mse",
    )
    start = regression_mae(model, features, target)
    model.fit(features, target, epochs=400, batch_size=12, verbose=0)
    finish = regression_mae(model, features, target)
    assert finish < start * 0.1


def test_saved_keras_model_predicts_the_same(tmp_path):
    model = build_bike_model(in_features=3, hidden=8, n_layers=2)
    features = np.random.default_rng(1).normal(size=(5, 3)).astype(np.float32)
    path = tmp_path / "bike.keras"
    model.save(path)
    loaded = keras.saving.load_model(path)
    np.testing.assert_allclose(
        np.asarray(model.predict(features, verbose=0)),
        np.asarray(loaded.predict(features, verbose=0)),
    )


def test_best_weights_are_restored():
    generator = np.random.default_rng(2)
    x_train = generator.normal(size=(64, 4)).astype(np.float32)
    y_train = (x_train[:, 0] * 3).astype(np.float32)
    x_val = generator.normal(size=(32, 4)).astype(np.float32)
    y_val = np.zeros(32, dtype=np.float32)
    model, rows, best_epoch, _seconds = fit_keras(
        x_train,
        y_train,
        x_val,
        y_val,
        seed=2,
        epochs=20,
        hidden=32,
        n_layers=2,
        learning_rate=0.2,
        batch_size=16,
    )
    best_row = min(rows, key=lambda row: row["val_loss"])
    last_row = rows[-1]
    assert best_epoch != last_row["epoch"]
    restored = regression_mae(model, x_val, y_val)
    assert abs(restored - best_row["val_mae"]) < 1e-4
    assert abs(restored - last_row["val_mae"]) > 1e-3


def test_same_seed_repeats_a_short_run():
    features = np.random.default_rng(3).normal(size=(24, 3)).astype(np.float32)
    target = features[:, 0].astype(np.float32)

    def weights(seed):
        model, _rows, _epoch, _seconds = fit_keras(
            features,
            target,
            features[:8],
            target[:8],
            seed=seed,
            epochs=2,
            hidden=4,
            n_layers=2,
            learning_rate=0.01,
            batch_size=8,
        )
        return model.get_weights()

    first = weights(11)
    second = weights(11)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
