from abc import ABC, abstractmethod
from typing import Dict, Union
import numpy as np
import xarray as xr

from constants import POWER, COAL, ASSET


class DeltaPosition(ABC):
    """
    Abstract base class for delta position estimation models.

    Parameters
    ----------
    fwds : xr.DataArray
        Forward price curves. Must contain an `ASSET` dimension
        (e.g., ["POWER", "COAL"]).
    y : np.ndarray
        Target variable (e.g., portfolio P&L) of shape (n_samples,).
    beta : xr.DataArray
        Sensitivity coefficients with the same `ASSET` dimension as `fwds`.

    Attributes
    ----------
    fwds : xr.DataArray
        Stored forward prices.
    y : np.ndarray
        Stored target variable.
    beta : xr.DataArray
        Stored beta coefficients.
    n_samples : int
        Number of samples inferred from `len(y)`.
    """

    def __init__(self, fwds: xr.DataArray, y: np.ndarray, beta: xr.DataArray) -> None:
        self.fwds: xr.DataArray = fwds
        self.x_fwd_power: np.ndarray = self.fwds.sel({ASSET: POWER}).values
        self.x_fwd_coal: np.ndarray = self.fwds.sel({ASSET: COAL}).values

        self.y: np.ndarray = y

        self.n_samples: int = len(y)

        self.beta: xr.DataArray = beta
        self.beta_power: np.ndarray = beta.sel({ASSET: POWER}).values
        self.beta_coal: np.ndarray = beta.sel({ASSET: COAL}).values

    @abstractmethod
    def compute(self) -> Dict[str, Union[np.ndarray, xr.DataArray, float]]:
        """
        Perform model-specific delta computation.

        Returns
        -------
        Dict[str, Union[np.ndarray, xr.DataArray, float]]
            Dictionary with:
            - KEY_PREDICTED : xr.DataArray
                Predicted deltas or model output.
            - KEY_RESIDUALS : np.ndarray
                Residuals (`y_true - y_pred`).
            - KEY_R2 : float
                R² coefficient of determination.
        """
        raise NotImplementedError