import os
import json
from pathlib import Path
from market_simulation.two_factor_model.two_factor_model import TwoFactorModelConfig

CONFIG_FOLDER_ENV = "CONFIG_FOLDER"  # Environment variable for config folder

# String constants for JSON keys
MODEL_1_KEY = "model_1"
MODEL_2_KEY = "model_2"
RHO_LONG_KEY = "rho_long"


class SpreadModelConfig:
    """
    Configuration container for a SpreadModel.

    Holds the configuration for two underlying two-factor forward models
    and the correlation between their long-term factors. Can be loaded from a JSON file.
    """

    def __init__(
        self,
        model1_config: TwoFactorModelConfig,
        model2_config: TwoFactorModelConfig,
        rho_long: float,
    ) -> None:
        """
        Parameters
        ----------
        model1_config : TwoFactorModelConfig
            Configuration of the first underlying two-factor model.
        model2_config : TwoFactorModelConfig
            Configuration of the second underlying two-factor model.
        rho_long : float
            Correlation between the long-term factors.
        """
        self.model1_config = model1_config
        self.model2_config = model2_config
        self.rho_long = rho_long

    @classmethod
    def from_json(cls, path: str) -> "SpreadModelConfig":
        """
        Load SpreadModel configuration from a JSON file.

        The JSON must contain:
            - MODEL_1_KEY: filename of first two-factor model config
            - MODEL_2_KEY: filename of second two-factor model config
            - RHO_LONG_KEY: correlation between long-term factors (mandatory)

        The model config files are looked up in the folder specified by
        the CONFIG_FOLDER environment variable.

        Parameters
        ----------
        path : str
            Path to the JSON file describing the spread model.

        Returns
        -------
        SpreadModelConfig
            An instance with the two-factor model configs loaded and rho_long set.

        Raises
        ------
        ValueError
            If CONFIG_FOLDER env variable is not set, or if required keys are missing.
        FileNotFoundError
            If the individual model config files do not exist.
        """
        env_path = os.getenv(CONFIG_FOLDER_ENV)
        if not env_path:
            raise ValueError(f"Environment variable {CONFIG_FOLDER_ENV} is not set.")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check required keys
        missing_keys = [
            k for k in [MODEL_1_KEY, MODEL_2_KEY, RHO_LONG_KEY] if k not in data
        ]
        if missing_keys:
            raise ValueError(f"Missing keys in SpreadModel JSON config: {missing_keys}")

        # Build full paths to the two-factor model config files
        model1_path = Path(env_path) / data[MODEL_1_KEY]
        model2_path = Path(env_path) / data[MODEL_2_KEY]

        # Load the individual TwoFactorModelConfig objects
        model1_config = TwoFactorModelConfig.from_json(str(model1_path))
        model2_config = TwoFactorModelConfig.from_json(str(model2_path))

        return cls(
            model1_config=model1_config,
            model2_config=model2_config,
            rho_long=float(data[RHO_LONG_KEY]),
        )
