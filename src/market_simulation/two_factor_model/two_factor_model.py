import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional

from src.market_simulation.market_helpers import extract_month_ahead, yfr
from src.market_simulation.constants import (
    SIMULATION_DAY,
    SIMULATION_PATH,
    DT,
    DELIVERY_START,
    FWD_CURVE,
)

from src.market_simulation.two_factor_model.two_factor_model_config import (
    TwoFactorModelConfig,
)


class TwoFactorForwardModel:
    def __init__(
        self,
        as_of_date: pd.Timestamp,
        simulation_days: pd.DatetimeIndex,
        params: Optional[TwoFactorModelConfig] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """
        Two-factor forward curve model.

        Parameters
        ----------
        as_of_date : pd.Timestamp
            Start of the simulation.
        simulation_days : pd.DatetimeIndex
            Simulation dates.
        params : TwoFactorModelParameters, optional
            Parameter container object.
        config_path : str, optional
            Path to JSON config file.
        """
        if simulation_days[0] != as_of_date:
            raise ValueError("as_of_date must equal the first simulation day")

        self.as_of_date = as_of_date
        self.simulation_days = simulation_days

        if params is None:
            if config_path is not None:
                params = TwoFactorModelConfig.from_json(config_path)
            else:
                raise ValueError("Either params or config_path must be provided.")

        # --- Two-factor params ---
        self.sigma_s = params.sigma_s
        self.kappa_s = params.kappa_s
        self.sigma_l = params.sigma_l
        self.kappa_l = params.kappa_l
        self.rho = params.rho

        # --- Day-ahead OU params ---
        self.beta = params.beta
        self.kappa = params.kappa

        # Cholesky decomposition for correlation
        corr_matrix = np.array([[1.0, self.rho], [self.rho, 1.0]])
        self.chol = np.linalg.cholesky(corr_matrix)

    def generate_dW(self, n_sims: int, n_steps: int) -> np.ndarray:
        """
        Generate independent standard normal shocks and apply correlation.

        Parameters
        ----------
        n_sims : int
            Number of simulation paths.
        n_steps : int
            Number of time steps (days).

        Returns
        -------
        np.ndarray
            Correlated shocks with shape (n_sims, n_steps-1, 2).
        """
        dW_indep = np.random.normal(size=(n_sims, n_steps - 1, 2))
        # Apply correlation vectorized
        return np.einsum("ab,sfb->sfa", self.chol, dW_indep)

    def log_var(self, maturity_date: pd.Timestamp) -> pd.Series:
        """
        Compute log-forward variance for each simulation day relative to maturity.
        """
        t_0 = 0.0
        t = np.array([yfr(self.as_of_date, d) for d in self.simulation_days])
        cap_t = yfr(self.as_of_date, maturity_date)

        # Integral-based variance contributions
        var_s = (self.sigma_s**2 / (2 * self.kappa_s)) * (
            np.exp(-2 * self.kappa_s * (cap_t - t))
            - np.exp(-2 * self.kappa_s * (cap_t - t_0))
        )
        var_l = (self.sigma_l**2 / (2 * self.kappa_l)) * (
            np.exp(-2 * self.kappa_l * (cap_t - t))
            - np.exp(-2 * self.kappa_l * (cap_t - t_0))
        )
        var_cross = (
            2 * self.rho * self.sigma_s * self.sigma_l / (self.kappa_s + self.kappa_l)
        ) * (
            np.exp(-(self.kappa_s + self.kappa_l) * (cap_t - t))
            - np.exp(-(self.kappa_s + self.kappa_l) * (cap_t - t_0))
        )

        total_var = var_s + var_l + var_cross

        # Set variance to NaN for simulation days >= maturity_date
        total_var = np.where(self.simulation_days >= maturity_date, np.nan, total_var)

        return pd.Series(total_var, index=self.simulation_days)

    def instant_fwd_vol(self, maturity_date: pd.Timestamp) -> pd.Series:
        """
        Compute instantaneous forward volatility for each simulation day relative to maturity.
        """
        tau = np.array([yfr(d, maturity_date) for d in self.simulation_days])

        vol_s = self.sigma_s * np.exp(-self.kappa_s * tau)
        vol_l = self.sigma_l * np.exp(-self.kappa_l * tau)
        vol_cross = 2 * self.rho * vol_s * vol_l

        total_vol = np.sqrt(vol_s**2 + vol_l**2 + vol_cross)

        # Set total_vol to NaN where tau < 0
        total_vol = np.where(self.simulation_days >= maturity_date, np.nan, total_vol)

        return pd.Series(total_vol, index=self.simulation_days)

    def var(self, fwd_0: float, maturity_date: pd.Timestamp) -> pd.Series:
        """
        Compute variance of F(t, T) for each simulation day relative to maturity.
        """
        log_vars = self.log_var(maturity_date)
        return fwd_0**2 * (np.exp(log_vars) - 1)

    def _simulate_day_ahead(
        self, month_ahead: xr.DataArray, n_sims: int
    ) -> xr.DataArray:
        """
        Simulate day-ahead prices by applying an OU process to month-ahead forwards.

        Parameters
        ----------
        month_ahead : xr.DataArray
            Month-ahead simulated series.
        n_sims : int
            Number of simulation paths.

        Returns
        -------
        xr.DataArray
            Day-ahead simulated series with same shape as month_ahead.
        """
        n_steps = len(self.simulation_days)
        dt = DT

        # OU process in log-space
        x = np.zeros((n_sims, n_steps))
        dw = np.random.normal(scale=np.sqrt(dt), size=(n_sims, n_steps - 1))

        for t in range(1, n_steps):
            x[:, t] = (
                x[:, t - 1] * np.exp(-self.kappa * dt)
                + self.beta
                * np.sqrt(1 - np.exp(-2 * self.kappa * dt))
                / np.sqrt(2 * self.kappa)
                * dw[:, t - 1]
            )

        exp_X = np.exp(x)
        day_ahead_vals = month_ahead.values * exp_X

        return xr.DataArray(
            day_ahead_vals,
            dims=[SIMULATION_PATH, SIMULATION_DAY],
            coords={
                SIMULATION_PATH: month_ahead[SIMULATION_PATH],
                SIMULATION_DAY: month_ahead[SIMULATION_DAY],
            },
            name="day_ahead",
        )

    def simulate(self, fwd_0: pd.Series, n_sims: int) -> xr.DataArray:
        """
        Simulate forward curves using internal simulation_days.
        """
        n_steps = len(self.simulation_days)
        dw = self.generate_dW(n_sims, n_steps) * np.sqrt(DT)

        delivery_start_dates = fwd_0.index
        n_delivery_starts = len(delivery_start_dates)

        fwds_array = np.zeros((n_steps, n_delivery_starts, n_sims))
        fwds_array[0, :, :] = np.tile(fwd_0.values[:, np.newaxis], (1, n_sims))

        # Precompute drift for all simulation days and maturities
        mu_df = pd.DataFrame(
            {
                maturity: -0.5 * self.log_var(maturity)
                for maturity in delivery_start_dates
            },
            index=self.simulation_days,
        )

        for m, maturity in enumerate(delivery_start_dates):
            mu_series = mu_df[maturity]

            for t in range(1, n_steps):
                current_date = self.simulation_days[t]
                if current_date >= maturity:
                    break

                tau = yfr(current_date, maturity)
                prev = fwds_array[t - 1, m, :]
                dw_s, dw_l = dw[:, t - 1].T
                mu = mu_series.loc[current_date]

                dF = prev * (
                    mu * DT
                    + self.sigma_s * np.exp(-self.kappa_s * tau) * dw_s
                    + self.sigma_l * np.exp(-self.kappa_l * tau) * dw_l
                )
                fwds_array[t, m, :] = prev + dF

        coords = {
            SIMULATION_PATH: np.arange(n_sims),
            SIMULATION_DAY: self.simulation_days,
            DELIVERY_START: delivery_start_dates,
        }

        fwds = xr.DataArray(
            np.transpose(fwds_array, (2, 0, 1)),
            dims=[SIMULATION_PATH, SIMULATION_DAY, DELIVERY_START],
            coords=coords,
            name=FWD_CURVE,
        )

        month_ahead = extract_month_ahead(fwds)

        day_ahead = None

        return fwds, month_ahead, day_ahead


if __name__ == "__main__":
    # Simulation setup
    as_of_date = pd.Timestamp("2025-09-13")
    n_days = 365
    simulation_days = pd.date_range(start=as_of_date, periods=n_days, freq="D")

    # Model initialization
    two_factor_model = TwoFactorForwardModel(
        as_of_date=as_of_date,
        simulation_days=simulation_days,
        config_path="power_2_factor_model_config.json",
    )

    # Forward curve setup
    n_rel_fwds = 12
    fwd_dates = pd.date_range(start=as_of_date, periods=n_rel_fwds, freq="MS")
    fwd_values = 50 + np.arange(n_rel_fwds)
    fwd_0 = pd.Series(fwd_values, index=fwd_dates)

    # Simulation
    n_sims = 1000
    fwds, month_ahead = two_factor_model.simulate(fwd_0=fwd_0, n_sims=n_sims)

    print(f"Simulated forwards shape: {fwds.shape}")
    print(f"Month-ahead derived series shape: {month_ahead.shape}")
