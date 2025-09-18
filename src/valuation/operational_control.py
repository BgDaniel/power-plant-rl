from enum import Enum

class OptimalControl(Enum):
    """Enum for the optimal control actions that the plant can take."""

    DO_NOTHING = 0  # Represents the action where no change occurs (idle state)
    RAMPING_UP = 1  # Represents the ramping up action
    RAMPING_DOWN = 2  # Represents the ramping down action
