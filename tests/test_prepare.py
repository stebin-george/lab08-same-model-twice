import numpy as np
import torch

from prepare import (
    NUMERIC_FEATURES,
    RentalDataset,
    fit_preparer,
    make_loader,
    split_frame,
    transform_features,
)


def test_dataset_length_and_float32_shapes():
    features = np.arange(6, dtype=np.float64).reshape(3, 2)
    target = np.array([1.0, 2.0, 3.0])
    dataset = RentalDataset(features, target)
    row, value = dataset[1]

    assert len(dataset) == 3
    assert row.dtype == torch.float32
    assert value.dtype == torch.float32
    assert tuple(row.shape) == (2,)
    assert tuple(value.shape) == ()


def test_splits_have_expected_sizes_and_do_not_overlap(rentals):
    train, validation, test = split_frame(rentals)

    assert (len(train), len(validation), len(test)) == (3600, 1200, 1200)
    assert len(train) + len(validation) + len(test) == len(rentals)
    train_index = set(train.index)
    validation_index = set(validation.index)
    test_index = set(test.index)
    assert train_index.isdisjoint(validation_index)
    assert train_index.isdisjoint(test_index)
    assert validation_index.isdisjoint(test_index)
    assert train_index | validation_index | test_index == set(rentals.index)


def test_scaler_is_fitted_on_training_rows_only():
    from bikes_data import make_rentals

    frame = make_rentals(n=500, seed=3)
    train, validation, _test = split_frame(frame, seed=3)
    scaler, encoder = fit_preparer(train)
    trained = transform_features(train, scaler, encoder)
    held_out = transform_features(validation, scaler, encoder)
    numeric_count = len(NUMERIC_FEATURES)

    assert np.allclose(trained[:, :numeric_count].mean(axis=0), 0, atol=1e-6)
    assert not np.allclose(held_out[:, :numeric_count].mean(axis=0), 0, atol=1e-3)
    assert np.allclose(scaler.mean_, train[NUMERIC_FEATURES].mean().to_numpy())
    assert not np.allclose(scaler.mean_, frame[NUMERIC_FEATURES].mean().to_numpy())
    assert encoder.n_features_in_ == 1


def _batch_order(loader):
    order = []
    for features, _target in loader:
        order.extend(features[:, 0].tolist())
    return order


def test_training_loader_shuffles_and_repeats_with_the_seed():
    features = np.arange(40, dtype=np.float32).reshape(40, 1)
    target = np.arange(40, dtype=np.float32)
    dataset = RentalDataset(features, target)

    first = _batch_order(make_loader(dataset, batch_size=10, shuffle=True, seed=7))
    second = _batch_order(make_loader(dataset, batch_size=10, shuffle=True, seed=7))

    assert first == second
    assert first != list(range(40))
