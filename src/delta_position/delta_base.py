from abc import ABC, abstractmethod
import numpy as np
import xarray as xr


from delta_position.delta_position_result import DeltaPositionResult


class DeltaBase(ABC):
    """
    Abstract base class for delta (hedge) estimation models.

    Implementations must be stateless and perform the full computation
    inside the `delta` method.
    """

    @abstractmethod
    def delta(
        self,
        t: float,
        fwds: xr.DataArray,
        y: np.ndarray,
        beta: xr.DataArray,
        efficiency: float,
    ) -> DeltaPositionResult:
        """
        Compute hedge deltas for a given set of inputs.

        Parameters
        ----------
        t: float
            Simulation time.
        fwds : xr.DataArray
            Forward prices with dimension ASSET.
        y : np.ndarray
            Target variable of shape (n_samples,).
        beta : xr.DataArray
            Asset betas with dimension ASSET.
        efficiency : float
            Efficiency factor used in spread calculation.

        Returns
        -------
        DeltaPositionResult
            Regression result including delta positions, predictions,
            residuals, and R².
        """
        raise NotImplementedError
