"""The Lab 07 network, built again in Keras.

A Keras Dense kernel is stored as (inputs, outputs). A PyTorch weight is
stored as (outputs, inputs). Copying weights has to transpose.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import keras
from torch import nn


def build_bike_model(in_features: int, hidden: int = 64, n_layers: int = 2):
    """Same shape as BikeNet: n hidden ReLU layers, then one count."""
    if n_layers < 2:
        raise ValueError("n_layers must be at least 2.")
    layers: list[keras.layers.Layer] = [keras.Input(shape=(in_features,))]
    for _ in range(n_layers):
        layers.append(keras.layers.Dense(hidden, activation="relu"))
    layers.append(keras.layers.Dense(1))
    return keras.Sequential(layers)


def copy_torch_weights(torch_model, keras_model) -> None:
    """Copy a PyTorch network's weights into the Keras network."""
    torch_layers = [
        module for module in torch_model.net if isinstance(module, nn.Linear)
    ]
    keras_layers = [
        layer for layer in keras_model.layers if isinstance(layer, keras.layers.Dense)
    ]
    if len(torch_layers) != len(keras_layers):
        raise ValueError("The two networks do not have the same layers.")
    for source, dest in zip(torch_layers, keras_layers, strict=True):
        kernel = source.weight.detach().cpu().numpy().T
        bias = source.bias.detach().cpu().numpy()
        dest.set_weights([kernel, bias])
