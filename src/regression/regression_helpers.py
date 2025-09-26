import numpy as np
from typing import Optional, Dict, Tuple

from constants import KEY_RESIDUALS, KEY_PREDICTED, KEY_R2

EPS = 1e-5


def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination) between predictions and target.

    Parameters
    ----------
    y_true : np.ndarray
        Actual/target values.
    y_pred : np.ndarray
        Predicted/regressed values.

    Returns
    -------
    float
        R² score. Ranges from -∞ to 1.0 (1.0 means perfect fit).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


def check_degenerate_case(
    x: np.ndarray, y: np.ndarray
) -> Tuple[bool, Optional[Dict[str, np.ndarray]]]:
    """
    Check if all features are nearly constant across samples.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples, n_features)
        Input feature matrix.
    y : np.ndarray, shape (n_samples,)
        Target vector.

    Returns
    -------
    Tuple[bool, Optional[Dict[str, np.ndarray]]]
        - is_degenerate : True if features are nearly constant, False otherwise.
        - result : dictionary with predicted, r2, residuals, condition_number if degenerate, else None.
    """
    x_range = x.max(axis=0) - x.min(axis=0)
    if np.all(x_range < EPS):
        y_pred = np.full_like(y, np.mean(y))
        residuals = y - y_pred
        result = {
            KEY_PREDICTED: y_pred,
            KEY_R2: 0.0,
            KEY_RESIDUALS: residuals
        }
        return True, result

    return False, None
