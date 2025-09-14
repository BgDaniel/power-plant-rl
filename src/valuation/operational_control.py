from enum import Enum

class OptimalControl(Enum):
    """Enum for the optimal control actions that the plant can take."""

    RAMPING_UP = 1  # Represents the ramping up action
    RAMPING_DOWN = 2  # Represents the ramping down action
    DO_NOTHING = 3  # Represents the action where no change occurs (idle state)