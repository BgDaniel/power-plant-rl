from dataclasses import dataclass
from market_simulation.config.base_config import BaseConfig


@dataclass(frozen=True)
class PowerPlantConfig(BaseConfig):
    """
    Configuration container for power plant operational parameters.

    Attributes
    ----------
    operation_costs : float
        Fixed operational costs of the power plant.
    alpha : float
        Coefficient for operational cost factor.
    ramping_up_costs : float
        Costs associated with ramping up operations.
    ramping_down_costs : float
        Costs associated with ramping down operations.
    n_days_ramping_up : int
        Number of days required to ramp up operations.
    n_days_ramping_down : int
        Number of days required to ramp down operations.
    idle_costs : float
        Costs associated with being idle.
    k : int
        Polynomial degree for regression (default is 3).
    """

    operation_costs: float
    alpha: float
    ramping_up_costs: float
    ramping_down_costs: float
    n_days_ramping_up: int
    n_days_ramping_down: int
    idle_costs: float
    k: int = 3  # default polynomial regression degree
