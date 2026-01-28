import torch
from torch.utils.data import TensorDataset, DataLoader
from nn_delta.nn_delta_model import NNDeltaModel
from nn_delta.sample_generator import SampleGenerator
from nn_delta.feature_builder import FeatureBuilder
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
import numpy as np


def train_nn(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dim: int = 64,
    batch_size: int = 128,
    epochs: int = 20,
    lr: float = 1e-3,
) -> NNDeltaModel:
    """
    Train a simple feed-forward NN to predict delta_power and delta_coal.

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    y : np.ndarray
        Target deltas of shape (n_samples, 2).
    hidden_dim : int
        Number of hidden units in each hidden layer.
    batch_size : int
        Training batch size.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate for Adam optimizer.

    Returns
    -------
    NNDeltaModel
        Trained neural network model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = NNDeltaModel(input_dim=X.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(dataset):.6f}")

    return model


if __name__ == "__main__":
    import pandas as pd
    import xarray as xr
    from valuation.power_plant.power_plant import PowerPlant

    # ----------------------------
    # Simulation setup (dummy data)
    # ----------------------------
    asset_days = pd.date_range("2026-01-01", periods=5)
    power_plant = PowerPlant(asset_days=asset_days, efficiency=0.4)

    n_sims = 10
    simulation_days = pd.date_range("2026-01-01", periods=5)

    # Dummy forwards & spots
    power_fwd = xr.DataArray(
        np.random.rand(n_sims, len(simulation_days), 1),
        dims=["simulation_path", "simulation_day", "delivery_start"],
        coords={
            "simulation_path": np.arange(n_sims),
            "simulation_day": simulation_days,
            "delivery_start": [simulation_days[0]],
        },
    )
    coal_fwd = xr.DataArray(
        np.random.rand(n_sims, len(simulation_days), 1),
        dims=["simulation_path", "simulation_day", "delivery_start"],
        coords={
            "simulation_path": np.arange(n_sims),
            "simulation_day": simulation_days,
            "delivery_start": [simulation_days[0]],
        },
    )
    power_spot = xr.DataArray(
        np.random.rand(n_sims, len(simulation_days)),
        dims=["simulation_path", "simulation_day"],
        coords={"simulation_path": np.arange(n_sims), "simulation_day": simulation_days},
    )
    coal_spot = xr.DataArray(
        np.random.rand(n_sims, len(simulation_days)),
        dims=["simulation_path", "simulation_day"],
        coords={"simulation_path": np.arange(n_sims), "simulation_day": simulation_days},
    )

    # ----------------------------
    # Generate training samples
    # ----------------------------
    generator = SampleGenerator(power_plant, n_sims=n_sims, efficiency=0.4)
    X, y = generator.generate(
        simulation_days,
        power_fwd,
        coal_fwd,
        power_spot,
        coal_spot,
        beta_power=np.ones(n_sims),
        beta_coal=np.ones(n_sims),
    )

    print(f"Generated sample shapes: X={X.shape}, y={y.shape}")

    # ----------------------------
    # Train neural network
    # ----------------------------
    model = train_nn(X, y, hidden_dim=16, batch_size=4, epochs=3, lr=1e-3)
    print("Training complete.")
