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


class NNDelta:
    """
    Neural network-based hedge delta estimation using logit outputs.

    Attributes:
        fwds (xr.DataArray): Forward prices of underlying assets.
        y (np.ndarray): Target values for regression.
        beta_power (np.ndarray): Beta exposure for power.
        beta_coal (np.ndarray): Beta exposure for coal.
        n_samples (int): Number of samples.
        X (np.ndarray): Feature matrix (power, coal, spread).
        X_scaled (np.ndarray): Scaled feature matrix [-1,1].
        nn (BaseNNRegressor): Neural network model.
    """

    def __init__(
        self,
        fwds: xr.DataArray,
        y: np.ndarray,
        beta: xr.DataArray,
        hidden_layers: list[int] = [32, 32],
    ) -> None:
        self.fwds: xr.DataArray = fwds
        self.y: np.ndarray = y
        self.beta_power: np.ndarray = beta.sel(asset="power").values
        self.beta_coal: np.ndarray = beta.sel(asset="coal").values
        self.n_samples: int = len(y)

        # Features: power, coal, spread
        self.x_power: np.ndarray = self.fwds.sel(asset="power").values
        self.x_coal: np.ndarray = self.fwds.sel(asset="coal").values
        self.x_spread: np.ndarray = self.x_power - self.x_coal
        self.X: np.ndarray = np.column_stack([self.x_power, self.x_coal, self.x_spread])

        # Neural network with logit outputs
        self.nn: BaseNNRegressor = BaseNNRegressor(
            input_dim=self.X.shape[1],
            hidden_layers=hidden_layers,
            output_activation=nn.Sigmoid,
        )

    def scale_features(self) -> None:
        """
        Scale input features to [-1, 1].

        Returns:
            None
        """
        self.X_scaled: np.ndarray = (
            2 * (self.X - self.X.min(axis=0)) / (self.X.ptp(axis=0) + 1e-8) - 1
        )

    def compute(self, epochs: int = 500, lr: float = 1e-3) -> Dict[str, Any]:
        """
        Train the neural network and compute delta positions.

        Args:
            epochs (int): Number of training epochs.
            lr (float): Learning rate.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - KEY_DELTA_POSITION: xr.DataArray of delta positions.
                - KEY_PREDICTED: Predicted regression values (np.ndarray).
                - KEY_RESIDUALS: Residuals of the regression (np.ndarray).
                - KEY_R2: R² score (float).
        """
        self.scale_features()
        self.nn.train_model(self.X_scaled, self.y, epochs=epochs, lr=lr)
        y_pred_nn: np.ndarray = self.nn.predict(self.X_scaled)  # shape (n_samples, 2)

        # Scale NN outputs by beta exposures
        delta_positions: xr.DataArray = xr.DataArray(
            np.column_stack(
                [y_pred_nn[:, 0] * self.beta_power, y_pred_nn[:, 1] * self.beta_coal]
            ),
            dims=[SIMULATION_PATH, ASSET],
            coords={
                ASSET: ["power", "coal"],
                SIMULATION_PATH: np.arange(self.n_samples),
            },
            name=KEY_DELTA_POSITION,
        )

        # Predicted as linear combination
        y_pred: np.ndarray = (
            y_pred_nn[:, 0] * self.beta_power + y_pred_nn[:, 1] * self.beta_coal
        )
        residuals: np.ndarray = self.y - y_pred
        r2: float = 1 - np.sum(residuals**2) / np.sum((self.y - np.mean(self.y)) ** 2)

        return {
            KEY_DELTA_POSITION: delta_positions,
            KEY_PREDICTED: y_pred,
            KEY_RESIDUALS: residuals,
            KEY_R2: r2,
        }
