from dataclasses import dataclass
from typing import Optional
import numpy as np
import xarray as xr


@dataclass(frozen=True)
class DeltaPositionResult:
    """
    Container for results of a delta (hedge) regression.

    Attributes
    ----------
    delta_position : xr.DataArray
        Delta positions per simulation path and asset.
        Dimensions: [SIMULATION_PATH, ASSET].
        Coordinates: ASSET = [POWER, COAL].

    predicted : np.ndarray
        Predicted values of the target variable y.
        Shape: (n_samples,).

    residuals : np.ndarray
        Regression residuals (y - predicted).
        Shape: (n_samples,).

    r_squared : Optional[float]
        Coefficient of determination (R²) of the regression.
        May be NaN or None in degenerate cases.
    """

    delta_positions: xr.DataArray
    predicted: np.ndarray
    residuals: np.ndarray
    r_squared: Optional[float]
