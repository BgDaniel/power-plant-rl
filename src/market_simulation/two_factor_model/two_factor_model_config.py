import os
import json
from dataclasses import dataclass
from pathlib import Path

CONFIG_FOLDER_ENV = "CONFIG_FOLDER"  # name of the environment variable
CONFIG_FOLDER = "config"


@dataclass(frozen=True)
class TwoFactorModelConfig:
    sigma_s: float
    kappa_s: float
    sigma_l: float
    kappa_l: float
    rho: float = 0.0

    @classmethod
    def from_json(cls, config_file_name: str) -> "TwoFactorModelConfig":
        """
        Load model parameters from a JSON configuration file located
        in the folder specified by the CONFIG_FOLDER environment variable.

        Parameters
        ----------
        config_file_name : str
            Name of the JSON config file.

        Returns
        -------
        TwoFactorModelConfig
            Instance with parameters loaded from the file.

        Raises
        ------
        FileNotFoundError
            If the config file does not exist in the folder.
        ValueError
            If the environment variable CONFIG_FOLDER is not set.
        """
        env_path = os.getenv(CONFIG_FOLDER_ENV)
        if not env_path:
            raise ValueError(f"Environment variable {CONFIG_FOLDER_ENV} is not set.")

        config_file = Path(env_path) / config_file_name
        if not config_file.exists():
            raise FileNotFoundError(
                f"Config file '{config_file_name}' not found in {config_file.parent}"
            )

        with config_file.open("r", encoding="utf-8") as f:
            params = json.load(f)

        return cls(
            sigma_s=float(params["sigma_s"]),
            kappa_s=float(params["kappa_s"]),
            sigma_l=float(params["sigma_l"]),
            kappa_l=float(params["kappa_l"]),
            rho=float(params.get("rho", 0.0)),
        )
