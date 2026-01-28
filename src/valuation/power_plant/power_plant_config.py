import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml

# Environment variable for config folder
CONFIG_FOLDER_ENV = "CONFIG_FOLDER"
CONFIG_FOLDER = "config"


@dataclass(frozen=True)
class PowerPlantConfig:
    """
    Configuration container for a power plant.

    Attributes
    ----------
    operation_costs : float
        Fixed operational costs of the power plant.
    efficiency : float
        Coefficient for operational cost factor.
    ramping_up_costs : float
        Costs associated with ramping up operations.
    ramping_down_costs : float
        Costs associated with ramping down operations.
    idle_costs : float
        Costs associated with being idle.
    n_days_ramping_up : int
        Number of days required to ramp up operations.
    n_days_ramping_down : int
        Number of days required to ramp down operations.
    polynomial_degree : int
        Degree of polynomial_regression_delta used for regression calculations.
    """

    # Costs
    operation_costs: float
    efficiency: float
    ramping_up_costs: float
    ramping_down_costs: float
    idle_costs: float

    # Technical constraints
    n_days_ramping_up: int
    n_days_ramping_down: int
    efficiency: float


class PowerPlantConfigLoader:
    """
    Loader for power plant YAML configurations.

    Expects YAML structure:
    costs:
      operation_costs: ...
      ramping_up_costs: ...
      ramping_down_costs: ...
      idle_costs: ...
    technical_constraints:
      n_days_ramping_up: ...
      n_days_ramping_down: ...
      efficiency: ...
    """

    @staticmethod
    def from_yaml(path: str) -> PowerPlantConfig:
        """
        Load a PowerPlantConfig from a YAML file.

        Parameters
        ----------
        path : str
            Path to the YAML configuration file.

        Returns
        -------
        PowerPlantConfig
            The loaded configuration dataclass.
        """
        env_path = os.getenv(CONFIG_FOLDER_ENV, CONFIG_FOLDER)
        config_file = Path(env_path) / path

        if not config_file.exists():
            raise FileNotFoundError(f"Config file '{config_file}' does not exist.")

        with config_file.open("r", encoding="utf-8") as f:
            config: Dict[str, Any] = yaml.safe_load(f)

        costs = config.get("costs", {})
        tech = config.get("technical_constraints", {})

        return PowerPlantConfig(
            operation_costs=costs.get("operation_costs", 0.0),
            ramping_up_costs=costs.get("ramping_up_costs", 0.0),
            ramping_down_costs=costs.get("ramping_down_costs", 0.0),
            idle_costs=costs.get("idle_costs", 0.0),
            n_days_ramping_up=tech.get("n_days_ramping_up", 1),
            n_days_ramping_down=tech.get("n_days_ramping_down", 1),
            efficiency=costs.get("efficiency", 1.0),
        )
