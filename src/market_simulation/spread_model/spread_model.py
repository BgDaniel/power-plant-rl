import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict
from typing import Optional
from market_simulation.two_factor_model.two_factor_model import TwoFactorForwardModel
from market_simulation.two_factor_model.two_factor_model_config import (
    TwoFactorModelConfig,
)
from market_simulation.spread_model.spread_model_config import SpreadModelConfig

SIMULATION = "simulation"
DAY = "day"
REL_FWD = "relative_forward"


class SpreadModel:
    """
    Spread model consisting of two underlying two-factor forward models
    with a long-term correlation factor.
    """

    def __init__(
        self,
        as_of_date: pd.Timestamp,
        simulation_days: pd.DatetimeIndex,
        params_two_factor_power_model: Optional[TwoFactorModelConfig] = None,
        params_two_factors_coal_model: Optional[TwoFactorModelConfig] = None,
        rho_long: Optional[float] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the SpreadModel.

        Parameters
        ----------
        params_two_factor_power_model : TwoFactorModelConfig, optional
            Parameters for the first underlying model.
        params_two_factors_coal_model : TwoFactorModelConfig, optional
            Parameters for the second underlying model.
        rho_long : float, optional
            Long-term correlation between the two models.
        config_path : str, optional
            Path to a JSON SpreadModel configuration file. If provided, this
            overrides the individual parameter objects.
        """
        if simulation_days[0] != as_of_date:
            raise ValueError("as_of_date must equal the first simulation day")

        self.as_of_date = as_of_date
        self.simulation_days = simulation_days

        if config_path is not None:
            # Load config from JSON
            spread_config = SpreadModelConfig.from_json(config_path)
            params_two_factor_power_model = spread_config.model1_config
            params_two_factors_coal_model = spread_config.model2_config
            rho_long = spread_config.rho_long

        # Validate that all required parameters are provided
        if (
            params_two_factor_power_model is None
            or params_two_factors_coal_model is None
            or rho_long is None
        ):
            raise ValueError(
                "Either provide all individual TwoFactorModelConfig objects and rho_long, "
                "or provide a valid config_path."
            )

        # Instantiate underlying two-factor models
        self.two_factor_power_model = TwoFactorForwardModel(
            as_of_date, simulation_days, params=params_two_factor_power_model
        )
        self.two_factor_coal_model = TwoFactorForwardModel(
            as_of_date, simulation_days, params=params_two_factors_coal_model
        )
        self.rho_long = rho_long

    def _generate_dW_values(self, n_sims: int, n_steps: int) -> np.ndarray:
        dW_indep = np.random.normal(size=(n_sims, n_steps - 1, 4))

        rho_power = self.two_factor_power_model.rho
        rho_coal = self.two_factor_coal_model.rho
        corr_matrix = np.array(
            [
                [1.0,       rho_power,      0.0,        0.0],
                [rho_power, 1.0,            0.0,        self.rho_long],
                [0.0,       0.0,            1.0,        rho_coal],
                [0.0,       self.rho_long,  rho_coal,   1.0],
            ]
        )

        chol = np.linalg.cholesky(corr_matrix)
        dW_corr = np.einsum("ab,sfb->sfa", chol, dW_indep)
        return dW_corr

    def simulate(
        self,
        power_fwd_0: pd.Series,
        coal_fwd_0: pd.Series,
        n_sims: int,
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
        n_steps = len(self.simulation_days)
        dw = self._generate_dW_values(n_sims, n_steps)

        # override underlying models' generate_dW_values
        self.two_factor_power_model.generate_dW_values = lambda n_s, n_t: dw[
            :, :, :2
        ]
        self.two_factor_coal_model.generate_dW_values = lambda n_s, n_t: dw[
            :, :, 2:
        ]

        # simulate both models
        power_fwd, power_month_ahead, power_spot = (
            self.two_factor_power_model.simulate(fwd_0=power_fwd_0, n_sims=n_sims)
        )
        coal_fwd, coal_month_ahead, coal_day_spot = (
            self.two_factor_coal_model.simulate(fwd_0=coal_fwd_0, n_sims=n_sims)
        )

        return (
            power_fwd,
            power_month_ahead,
            power_spot,
            coal_fwd,
            coal_month_ahead,
            coal_day_spot,
        )
