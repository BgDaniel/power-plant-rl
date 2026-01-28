import numpy as np
import pandas as pd
from market_simulation.constants import DT
from numpy.polynomial.legendre import legval


def instant_fwd_vol(sample: pd.DataFrame) -> pd.Series:
    """
    Compute empirical instantaneous forward volatility from simulated forward paths.

    Parameters
    ----------
    sample : pd.DataFrame
        Simulated forward paths with shape (n_sims, n_days). Columns are time steps.

    Returns
    -------
    pd.Series
        Instantaneous forward volatilities for each time step (n_days - 1), indexed by columns.
    """
    log_fwds = np.log(sample)
    dlog = log_fwds.diff(axis=0)
    inst_fwd_vol = np.sqrt(dlog.var(axis=1, ddof=1) / DT)
    return inst_fwd_vol


def log_var(sample: pd.DataFrame) -> pd.Series:
    """
    Compute empirical log-variance across simulations.

    Parameters
    ----------
    sample : pd.DataFrame
        Simulated forward paths, shape (n_sims, n_days).

    Returns
    -------
    pd.Series
        Log-variance for each time step, indexed by columns.
    """
    log_fwds = np.log(sample)
    return log_fwds.var(axis=1, ddof=1)



def shifted_legendre_basis_dynamic(x: float, degree: int, x_max: float) -> np.ndarray:
    """
    Evaluate shifted Legendre polynomials on [0, x_max].

    Args:
        x: Value to evaluate.
        degree: Maximum polynomial_regression_delta degree.
        x_max: Upper bound of the interval.

    Returns:
        Array of polynomial_regression_delta values [P0(x), ..., P_degree(x)].
    """
    y = 2 * np.array(x) / max(x_max, 1e-6) - 1
    coeffs = np.eye(degree + 1)
    return legval(y, coeffs.T)
