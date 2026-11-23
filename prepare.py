"""Split the rental table and turn rows into tensors.

The split happens before any scaler or encoder is fitted. Both are fitted on
the training rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import TARGET

SEED = 2026
NUMERIC_FEATURES = ["hour", "temperature_c", "rain_mm", "wind_kph", "is_weekend"]
CATEGORY_FEATURES = ["station_type"]


def split_frame(frame: pd.DataFrame, seed: int = SEED):
    """Return train, validation and test rows: 60%, 20% and 20%."""
    indexes = np.arange(len(frame))
    train_index, holdout_index = train_test_split(
        indexes, test_size=0.4, random_state=seed, shuffle=True
    )
    val_index, test_index = train_test_split(
        holdout_index, test_size=0.5, random_state=seed, shuffle=True
    )
    return (
        frame.iloc[train_index].copy(),
        frame.iloc[val_index].copy(),
        frame.iloc[test_index].copy(),
    )


def fit_preparer(train_frame: pd.DataFrame):
    """Fit the scaler and the station encoder on training rows only."""
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler.fit(train_frame[NUMERIC_FEATURES])
    encoder.fit(train_frame[CATEGORY_FEATURES])
    return scaler, encoder


def transform_features(frame, scaler, encoder) -> np.ndarray:
    """Scale numbers and one-hot encode the station type."""
    numbers = scaler.transform(frame[NUMERIC_FEATURES])
    categories = encoder.transform(frame[CATEGORY_FEATURES])
    return np.hstack([numbers, categories]).astype(np.float32)


def target_values(frame: pd.DataFrame) -> np.ndarray:
    return frame[TARGET].to_numpy(dtype=np.float32)


class RentalDataset(Dataset):
    """One station-hour at a time, as float32 tensors."""

    def __init__(self, features: np.ndarray, target: np.ndarray) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.target = torch.as_tensor(target, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int):
        return self.features[index], self.target[index]


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int):
    """Shuffle training batches only. The generator makes that order repeatable."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
