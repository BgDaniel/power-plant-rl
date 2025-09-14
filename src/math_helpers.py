import numpy as np
import pandas as pd
from market_simulation.constants import DT
from numpy.polynomial.legendre import legval


def instantaneous_forward_volatility(sample: pd.DataFrame) -> pd.Series:
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
    dlog = log_fwds.diff(axis=1).iloc[:, 1:]  # differences along columns
    inst_var = (dlog / np.sqrt(DT)).var(axis=0, ddof=1)
    return inst_var


def log_variance(sample: pd.DataFrame) -> pd.Series:
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
    return log_fwds.var(axis=0, ddof=1)


def variance(sample: pd.DataFrame) -> pd.Series:
    """
    Compute empirical variance of forward levels across simulations.

    Parameters
    ----------
    sample : pd.DataFrame
        Simulated forward paths, shape (n_sims, n_days).

    Returns
    -------
    pd.Series
        Variance for each time step, indexed by columns.
    """
    return sample.var(axis=0, ddof=1)


def mean_forward(sample: pd.DataFrame) -> pd.Series:
    """
    Compute empirical mean of forward levels across simulations.

    Parameters
    ----------
    sample : pd.DataFrame
        Simulated forward paths, shape (n_sims, n_days).

    Returns
    -------
    pd.Series
        Mean for each time step, indexed by columns.
    """
    return sample.mean(axis=0)


def shifted_legendre_basis_dynamic(x: float, degree: int, x_max: float) -> np.ndarray:
    """
    Evaluate shifted Legendre polynomials on [0, x_max].

    Args:
        x: Value to evaluate.
        degree: Maximum polynomial degree.
        x_max: Upper bound of the interval.

    Returns:
        Array of polynomial values [P0(x), ..., P_degree(x)].
    """
    y = 2 * np.array(x) / max(x_max, 1e-6) - 1
    coeffs = np.eye(degree + 1)
    return legval(y, coeffs.T)
