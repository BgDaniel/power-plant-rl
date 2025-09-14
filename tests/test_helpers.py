import pandas as pd
import numpy as np


def assert_relative_difference(
    empirical: pd.Series,
    analytical: pd.Series,
    max_diff: float = 0.05,
    name: str = "Series",
) -> None:
    """
    Assert that the relative difference between two Pandas Series is below a tolerance.

    Parameters
    ----------
    empirical : pd.Series
        Empirical values from simulation.
    analytical : pd.Series
        Analytical or benchmark values.
    max_diff : float, default=0.05
        Maximum allowed relative difference.
    name : str, default="Series"
        Name for error messages.
    """
    # Restrict to common indices
    common_idx = empirical.index.intersection(analytical.index)
    emp_common = empirical.loc[common_idx]
    ana_common = analytical.loc[common_idx]

    # Compute relative difference safely
    rel_diff = np.abs(emp_common - ana_common) / (ana_common + 1e-12)

    max_rel_diff = np.nanmax(rel_diff)
    assert (
        max_rel_diff < max_diff
    ), f"Empirical {name} deviates too much: {max_rel_diff:.2%}"
