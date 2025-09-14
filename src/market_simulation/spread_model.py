import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional, Dict
from market_simulation.two_factor_model import TwoFactorForwardModel

SIMULATION = "simulation"
DAY = "day"
REL_FWD = "relative_forward"


class SpreadModel:
    def __init__(
        self,
        params_two_factors_1: Dict[str, float],
        params_two_factors_2: Dict[str, float],
        rho_long: float = 0.0,
    ) -> None:
        self.model1 = TwoFactorForwardModel(**params_two_factors_1)
        self.model2 = TwoFactorForwardModel(**params_two_factors_2)
        self.rho_long = rho_long

    def _generate_dW_values(self, n_sims: int, n_steps: int) -> np.ndarray:
        dW_indep = np.random.normal(size=(n_sims, n_steps - 1, 4))

        rho1 = self.model1.rho
        rho2 = self.model2.rho
        corr_matrix = np.array(
            [
                [1.0, rho1, 0.0, 0.0],
                [rho1, 1.0, 0.0, self.rho_long],
                [0.0, 0.0, 1.0, rho2],
                [0.0, self.rho_long, rho2, 1.0],
            ]
        )

        chol = np.linalg.cholesky(corr_matrix)
        dW_corr = np.einsum("ab,sfb->sfa", chol, dW_indep)
        return dW_corr

    def simulate(
        self,
        F0_1: pd.Series,
        F0_2: pd.Series,
        simulation_days: pd.DatetimeIndex,
        n_sims: int,
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
        n_steps = len(simulation_days)
        dW_values = self._generate_dW_values(n_sims, n_steps)

        # override underlying models' generate_dW_values
        self.model1.generate_dW_values = lambda n_s, n_t: dW_values[:, :, :2]
        self.model2.generate_dW_values = lambda n_s, n_t: dW_values[:, :, 2:]

        # simulate both models
        fwds_1, month_ahead_1 = self.model1.simulate(
            F0=F0_1, n_sims=n_sims, simulation_days=simulation_days
        )
        fwds_2, month_ahead_2 = self.model2.simulate(
            F0=F0_2, n_sims=n_sims, simulation_days=simulation_days
        )

        return fwds_1, month_ahead_1, fwds_2, month_ahead_2


# -------------------------------
# Run as script
# -------------------------------
if __name__ == "__main__":
    # Example parameters
    params1 = dict(
        mu=0.01, sigma_s=0.05, kappa_s=1.0, sigma_l=0.02, kappa_l=0.1, rho=0.3
    )
    params2 = dict(
        mu=0.02, sigma_s=0.04, kappa_s=0.8, sigma_l=0.03, kappa_l=0.05, rho=0.2
    )
    rho_long = 0.5

    model = SpreadModel(params1, params2, rho_long=rho_long)

    start_date = pd.Timestamp("2025-09-13")
    n_rel_fwds = 6
    fwd_dates = pd.date_range(start=start_date, periods=n_rel_fwds, freq="MS")
    F0_1 = pd.Series(data=50 + np.arange(n_rel_fwds), index=fwd_dates)
    F0_2 = pd.Series(data=60 + np.arange(n_rel_fwds), index=fwd_dates)

    simulation_days = pd.date_range(start=start_date, periods=30, freq="D")
    n_sims = 500

    fwds_1, month_ahead_1, fwds_2, month_ahead_2 = model.simulate(
        F0_1, F0_2, simulation_days, n_sims
    )

    print("Model 1 simulated forwards shape:", fwds_1.shape)
    print("Model 2 simulated forwards shape:", fwds_2.shape)
