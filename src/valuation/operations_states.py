from enum import Enum

from valuation.operational_control import OptimalControl


class OperationalState(Enum):
    """Enum for the operational states of the power plant."""

    IDLE = 0  # The plant is idle (not generating power).
    RAMPING_UP = 1  # The plant is ramping up to generate power.
    RAMPING_DOWN = 2  # The plant is ramping down from power generation.
    RUNNING = 3  # The plant is actively generating power.


def get_next_state(
    current_optimal_state: OperationalState, optimal_control: OptimalControl
) -> OperationalState:
    """
    Get the next operational state based on the current state and optimal control decision.

    Args:
        current_optimal_state (OperationalState): The current state of the power plant.
        optimal_control (OptimalControl): The optimal control decision for the next step.

    Returns:
        OperationalState: The next operational state after applying the optimal control decision.
    """
    if current_optimal_state == OperationalState.IDLE:
        if optimal_control == OptimalControl.RAMPING_UP:
            return OperationalState.RAMPING_UP
        else:
            return OperationalState.IDLE  # Do Nothing remains in IDLE

    elif current_optimal_state == OperationalState.RAMPING_UP:
        return OperationalState.RUNNING

    elif current_optimal_state == OperationalState.RUNNING:
        if optimal_control == OptimalControl.RAMPING_DOWN:
            return OperationalState.RAMPING_DOWN
        else:  # DO_NOTHING
            return OperationalState.RUNNING  # Stay running

    elif current_optimal_state == OperationalState.RAMPING_DOWN:
        return OperationalState.IDLE