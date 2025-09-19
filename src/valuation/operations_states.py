from enum import Enum
import numpy as np
import pandas as pd
import xarray as xr


class OperationalState(Enum):
    """Enum for the operational states of the power plant."""

    IDLE = 0  # The plant is idle (not generating power).
    RAMPING_UP = 1  # The plant is ramping up to generate power.
    RAMPING_DOWN = 2  # The plant is ramping down from power generation.
    RUNNING = 3  # The plant is actively generating power.
    UNDEFINED = 99  # Special marker for invalid transitions


# Transition matrix: rows = current state, columns = control
# Values = next state
NEXT_STATE_MATRIX = np.array(
    [
        # DO_NOTHING, RAMPING_UP, RAMPING_DOWN
        [
            OperationalState.IDLE,
            OperationalState.RAMPING_UP,
            OperationalState.UNDEFINED,
        ],  # IDLE
        [
            OperationalState.RUNNING,
            OperationalState.UNDEFINED,
            OperationalState.UNDEFINED,
        ],  # RAMPING_UP
        [
            OperationalState.IDLE,
            OperationalState.UNDEFINED,
            OperationalState.UNDEFINED,
        ],  # RAMPING_DOWN
        [
            OperationalState.RUNNING,
            OperationalState.UNDEFINED,
            OperationalState.RAMPING_DOWN,
        ],  # RUNNING
    ]
)


def get_next_state(prev_states: pd.Series, prev_controls: xr.DataArray) -> np.ndarray:
    """
    Vectorized computation of next operational states.

    Parameters
    ----------
    prev_states : pd.Series[OperationalState]
        Current operational states as enums. Indexed by simulation path.
    prev_controls : xr.DataArray
        Optimal control decisions as enums. One-dimensional, length n_paths.

    Returns
    -------
    np.ndarray
        Next operational states as integers. Shape: (n_paths,)
    """
    # Convert prev_states (Series of enums) to integers
    prev_states_int = prev_states.map(lambda s: s.value).to_numpy(dtype=int)

    # Convert prev_controls (DataArray of enums) to integers
    prev_controls_int = np.array(
        [ctrl.value for ctrl in prev_controls.values], dtype=int
    )

    next_states = NEXT_STATE_MATRIX[prev_states_int, prev_controls_int]

    # Check for invalid transitions
    if np.any(next_states == OperationalState.UNDEFINED):
        invalid_indices = np.where(next_states == OperationalState.UNDEFINED)[0]
        raise ValueError(
            f"Invalid state transition requested for paths: {invalid_indices}"
        )

    return next_states
