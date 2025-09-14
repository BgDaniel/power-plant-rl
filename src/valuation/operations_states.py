from enum import Enum

class OperationalState(Enum):
    """Enum for the operational states of the power plant."""

    IDLE = 0  # The plant is idle (not generating power).
    RAMPING_UP = 1  # The plant is ramping up to generate power.
    RAMPING_DOWN = 2  # The plant is ramping down from power generation.
    RUNNING = 3  # The plant is actively generating power.
