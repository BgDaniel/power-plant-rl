from typing import Dict, Any
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from constants import (
    SIMULATION_PATH,
    ASSET,
    KEY_DELTA_POSITION,
    KEY_PREDICTED,
    KEY_RESIDUALS,
    KEY_R2,
)


class BaseNNRegressor(nn.Module):
    """
    Base neural network regressor for hedge delta estimation.

    Args:
        input_dim (int): Number of input features.
        hidden_layers (list[int]): List of hidden layer sizes.
        activation: Activation function class (default: nn.ReLU).
        output_activation: Output activation function class (default: nn.Sigmoid for bounded outputs).

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass through the network.
        train_model(X: np.ndarray, y: np.ndarray, epochs: int, lr: float) -> float:
            Train the network using MSE loss.
        predict(X: np.ndarray) -> np.ndarray:
            Predict outputs for given inputs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int] = [32, 32],
        activation=nn.ReLU,
        output_activation=nn.Sigmoid,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))  # Two outputs: power and coal deltas
        layers.append(output_activation())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Output tensor of shape (n_samples, 2).
        """
        return self.model(x)

    def train_model(
        self, X: np.ndarray, y: np.ndarray, epochs: int = 500, lr: float = 1e-3
    ) -> float:
        """
        Train the neural network using mean squared error loss.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
            epochs (int): Number of training epochs.
            lr (float): Learning rate.

        Returns:
            float: Final training loss.
        """
        X_tensor = torch.from_numpy(X.astype(np.float32))
        y_tensor = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            y_pred = self(X_tensor)
            # Replicate y to match the two outputs
            y_target = torch.cat([y_tensor, y_tensor], dim=1)
            loss = criterion(y_pred, y_target)
            loss.backward()
            optimizer.step()
        return loss.item()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outputs for given input features.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted outputs of shape (n_samples, 2).
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X.astype(np.float32))
            return self(X_tensor).numpy()
