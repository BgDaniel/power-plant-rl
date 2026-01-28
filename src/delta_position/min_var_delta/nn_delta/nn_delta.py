from typing import Optional
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xarray as xr

from delta_position.delta_base import DeltaBase
from delta_position.delta_position_result import DeltaPositionResult
from delta_position.min_var_delta.nn_delta.nn_delta_model import NNDeltaModel


class NNDelta(DeltaBase):
    """
    Neural-network-based hedge delta estimator.

    Neural network input features (always 4):
        1. fwd_power: forward power price
        2. fwd_coal: forward coal price
        3. spread: fwd_power - efficiency * fwd_coal
        4. t: delivery month (float)

    Output is 2-dimensional:
        - delta_power
        - delta_coal

    Loss is minimum-variance:
        y_hat = beta_power * delta_power + beta_coal * delta_coal
    """

    def __init__(
        self,
        model: NNDeltaModel,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize NNDelta with a given neural network.

        Parameters
        ----------
        model : NNDeltaModel
            Pre-built PyTorch model to use for delta estimation. Must output 2 values.
        device : str or None, optional
            Torch device ("cpu" or "cuda"). If None, auto-select.

        Attributes
        ----------
        model : nn.Module
            Neural network used for delta estimation.
        device : str
            Torch device in use.
        trained : bool
            Flag indicating whether the model is trained.
        """
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module = model.to(self.device)
        self.trained: bool = False

    def delta(
        self,
        t: float,
        fwds: xr.DataArray,
        y: np.ndarray,
        beta: xr.DataArray,
        efficiency: float,
    ) -> DeltaPositionResult:
        """
        Compute hedge deltas for the given simulation using the trained NN.

        Parameters
        ----------
        t : float
            Delivery or simulation time.
        fwds : xr.DataArray
            Forward prices, dimension should include "asset".
        y : np.ndarray
            Target cashflows of shape (n_samples,).
        beta : xr.DataArray
            Beta exposures, dimension should include "asset".
        efficiency : float
            Efficiency factor for spread calculation.

        Returns
        -------
        DeltaPositionResult
            Object containing:
            - delta_position : xr.DataArray (shape: n_samples x 2)
            - predicted : np.ndarray, predicted cashflows
            - residuals : np.ndarray, residuals
            - r2 : Optional[float], not computed here
        """
        if not self.trained:
            raise RuntimeError("NNDelta model must be trained before calling delta()")

        # Extract forward prices
        fwd_power = fwds.sel(asset="POWER").values
        fwd_coal = fwds.sel(asset="COAL").values
        spread = fwd_power - efficiency * fwd_coal

        # Build NN features
        features = np.column_stack(
            [
                fwd_power,
                fwd_coal,
                spread,
                beta.sel(asset="POWER").values,
                beta.sel(asset="COAL").values,
                np.full_like(fwd_power, t),
            ]
        )

        X_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            delta_pred = self.model(X_tensor).cpu().numpy()

        delta_power = delta_pred[:, 0]
        delta_coal = delta_pred[:, 1]
        y_hat = (
            beta.sel(asset="POWER").values * delta_power
            + beta.sel(asset="COAL").values * delta_coal
        )
        residuals = y - y_hat

        # Build xarray DataArray for delta positions
        delta_positions = xr.DataArray(
            np.stack([delta_power, delta_coal], axis=1),
            dims=[fwds.dims[0], "asset"],
            coords={
                fwds.dims[0]: fwds.coords[fwds.dims[0]].values,
                "asset": ["POWER", "COAL"],
            },
            name="delta_position",
        )

        return DeltaPositionResult(
            delta_position=delta_positions,
            predicted=y_hat,
            residuals=residuals,
            r2=None,
        )
