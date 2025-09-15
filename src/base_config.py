import os
import json
from pathlib import Path
from dataclasses import dataclass, fields, MISSING, Field
from typing import Type, TypeVar, Dict, Any, cast

# Environment variable for overriding config folder
CONFIG_FOLDER_ENV = "CONFIG_FOLDER"
CONFIG_FOLDER = "config"

# Generic type variable restricted to BaseConfig subclasses
T = TypeVar("T", bound="BaseConfig")


class BaseConfig:
    """
    Abstract base class for configuration dataclasses.

    Provides generic utilities to:
    - Load configurations from JSON files.
    - Map JSON keys to dataclass fields.
    - Apply default values when keys are missing.
    """

    @classmethod
    def from_json(cls: Type[T], path_to_config: str) -> T:
        """
        Load a configuration dataclass from a JSON file.

        The function looks for the file in a directory specified by the
        CONFIG_FOLDER_ENV environment variable, or defaults to 'config/'.

        Args:
            path_to_config (str): Relative path to the JSON config file.

        Returns:
            T: An instance of the dataclass subclass (e.g., TwoFactorModelConfig).

        Raises:
            FileNotFoundError: If the JSON file does not exist.
            KeyError: If required fields are missing from the config.
            ValueError: If type casting of fields fails.
        """
        env_path: str = os.getenv(CONFIG_FOLDER_ENV, CONFIG_FOLDER)
        config_file: Path = Path(env_path) / path_to_config

        if not config_file.exists():
            raise FileNotFoundError(
                f"Config file '{path_to_config}' not found in {config_file.parent}"
            )

        with config_file.open("r", encoding="utf-8") as f:
            params: Dict[str, Any] = json.load(f)

        return cls._from_dict(params)

    @classmethod
    def _from_dict(cls: Type[T], params: Dict[str, Any]) -> T:
        """
        Map a dictionary of parameters to the dataclass fields.

        Args:
            params (Dict[str, Any]): Dictionary containing field values.

        Returns:
            T: An instance of the dataclass subclass.

        Raises:
            KeyError: If required fields are missing from the config.
            ValueError: If type casting of a field fails.
        """
        init_args: Dict[str, Any] = {}

        for f in fields(cls):  # type: ignore[arg-type]
            if f.name in params:
                try:
                    # Attempt type casting to match field type
                    init_args[f.name] = f.type(params[f.name])  # type: ignore
                except Exception as e:
                    raise ValueError(
                        f"Failed to cast field '{f.name}' to {f.type}: {params[f.name]}"
                    ) from e
            elif f.default is not MISSING:
                init_args[f.name] = f.default
            elif f.default_factory is not MISSING:  # type: ignore
                init_args[f.name] = f.default_factory()  # type: ignore
            else:
                raise KeyError(f"Missing required config field '{f.name}'")

        return cls(**init_args)  # type: ignore
