import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional, Tuple
from numba import njit


from src.market_simulation.two_factor_model.simulation_caching import cache_simulation
from src.market_simulation.market_helpers import extract_month_ahead, yfr
from src.market_simulation.constants import (
    SIMULATION_DAY,
    SIMULATION_PATH,
    DT,
    DELIVERY_START,
    FWD_CURVE,
)
from src.market_simulation.two_factor_model.two_factor_model_config import TwoFactorModelConfig


class TwoFactorForwardModel:
    """
    Two-factor forward curve model for energy markets.

    Implements a two-factor model for forward curves with short-term and long-term factors,
    as well as an OU-based day-ahead spot price process.
    """

    def __init__(
            self,
            as_of_date: pd.Timestamp,
            simulation_days: pd.DatetimeIndex,
            params: Optional[TwoFactorModelConfig] = None,
            config_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the TwoFactorForwardModel.

        Parameters
        ----------
        as_of_date : pd.Timestamp
            Start date of the simulation.
        simulation_days : pd.DatetimeIndex
            Array of simulation dates.
        params : TwoFactorModelConfig, optional
            Model parameters.
        config_path : str, optional
            Path to JSON configuration file.
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
        self.sigma = params.sigma

        # Cholesky decomposition for correlation
        corr_matrix = np.array([[1.0, self.rho], [self.rho, 1.0]])
        self.chol = np.linalg.cholesky(corr_matrix)

    def generate_dW(self, n_sims: int, n_steps: int) -> np.ndarray:
        """
        Generate correlated standard normal shocks.

        Parameters
        ----------
        n_sims : int
            Number of simulation paths.
        n_steps : int
            Number of time steps (days).

        Returns
        -------
        np.ndarray
            Correlated shocks with shape (n_sims, n_steps-1, 2)
            corresponding to the short- and long-term factors.
        """
        dW_indep = np.random.normal(size=(n_sims, n_steps - 1, 2))
        return np.einsum("ab,sfb->sfa", self.chol, dW_indep)

    def log_var(self, maturity_date: pd.Timestamp) -> pd.Series:
        """
        Compute log-forward variance for each simulation day relative to a given maturity.

        Parameters
        ----------
        maturity_date : pd.Timestamp
            The maturity date for which the variance is computed.

        Returns
        -------
        pd.Series
            Log-forward variance for each simulation day.
        """
        t_0 = 0.0
        t = np.array([yfr(self.as_of_date, d) for d in self.simulation_days])
        cap_t = yfr(self.as_of_date, maturity_date)

        var_s = (self.sigma_s ** 2 / (2 * self.kappa_s)) * (
                np.exp(-2 * self.kappa_s * (cap_t - t))
                - np.exp(-2 * self.kappa_s * (cap_t - t_0))
        )
        var_l = (self.sigma_l ** 2 / (2 * self.kappa_l)) * (
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
        total_var = np.where(self.simulation_days >= maturity_date, np.nan, total_var)
        return pd.Series(total_var, index=self.simulation_days)

    def instant_fwd_vol(self, maturity_date: pd.Timestamp) -> pd.Series:
        """
        Compute instantaneous forward volatility for each simulation day relative to a given maturity.

        Parameters
        ----------
        maturity_date : pd.Timestamp
            The maturity date for which volatility is computed.

        Returns
        -------
        pd.Series
            Instantaneous forward volatility for each simulation day.
        """
        tau = np.array([yfr(d, maturity_date) for d in self.simulation_days])
        vol_s = self.sigma_s * np.exp(-self.kappa_s * tau)
        vol_l = self.sigma_l * np.exp(-self.kappa_l * tau)
        vol_cross = 2 * self.rho * vol_s * vol_l
        total_vol = np.sqrt(vol_s ** 2 + vol_l ** 2 + vol_cross)
        total_vol = np.where(self.simulation_days >= maturity_date, np.nan, total_vol)
        return pd.Series(total_vol, index=self.simulation_days)

    def var(self, fwd_0: float, maturity_date: pd.Timestamp) -> pd.Series:
        """
        Compute variance of F(t,T) for each simulation day.

        Parameters
        ----------
        fwd_0 : float
            Initial forward price.
        maturity_date : pd.Timestamp
            The maturity date for which variance is computed.

        Returns
        -------
        pd.Series
            Variance of F(t,T) for each simulation day.
        """
        log_vars = self.log_var(maturity_date)
        return fwd_0 ** 2 * (np.exp(log_vars) - 1)

    def var_ou(self, sim_dates: pd.DatetimeIndex) -> pd.Series:
        """
        Compute the variance of the OU day-ahead process.

        Parameters
        ----------
        sim_dates : pd.DatetimeIndex
            Dates for which variance is computed.

        Returns
        -------
        pd.Series
            OU variance indexed by simulation dates.
        """
        t = np.array([yfr(self.as_of_date, d) for d in sim_dates])
        var = (self.sigma ** 2 / (2 * self.beta)) * (1 - np.exp(-2 * self.beta * t))
        return pd.Series(var, index=sim_dates)

    def day_ahead_var(self, sim_dates: pd.DatetimeIndex) -> pd.Series:
        """
        Compute variance of exponential OU day-ahead process.

        Parameters
        ----------
        sim_dates : pd.DatetimeIndex
            Dates for which variance is computed.

        Returns
        -------
        pd.Series
            Variance of the day-ahead process.
        """
        ou_var = self.var_ou(sim_dates)
        variances = np.exp(ou_var) - 1
        return pd.Series(variances, index=sim_dates)

    def _simulate_spot_prices(
            self, month_ahead: xr.DataArray, n_sims: int
    ) -> pd.DataFrame:
        """
        Simulate day-ahead prices by applying an OU process to month-ahead forwards.

        Parameters
        ----------
        month_ahead : xr.DataArray
            Month-ahead forward series.
        n_sims : int
            Number of simulation paths.

        Returns
        -------
        pd.DataFrame
            Day-ahead simulated series as a DataFrame.
        """
        n_steps = len(self.simulation_days)
        x = np.zeros((n_steps, n_sims))
        dw = np.random.normal(size=(n_steps - 1, n_sims)) * np.sqrt(DT)

        for t in range(1, n_steps):
            x[t, :] = x[t - 1, :] - self.beta * x[t - 1, :] * DT + self.sigma * dw[t - 1, :]

        var_ou_array = self.var_ou(self.simulation_days).values[:, np.newaxis]
        day_ahead = np.exp(-var_ou_array / 2 + x)
        spot_price_vals = month_ahead.values * day_ahead
        return pd.DataFrame(index=month_ahead.index, data=spot_price_vals)

    @staticmethod
    @njit
    def _simulate_forward_loop(fwds_array: np.ndarray, mu_array: np.ndarray,
                               sigma_s: float, kappa_s: float,
                               sigma_l: float, kappa_l: float,
                               dw: np.ndarray, tau_array: np.ndarray) -> np.ndarray:
        """
        Numba-accelerated forward simulation loop.

        Parameters
        ----------
        fwds_array : np.ndarray
            Array to store forward values (n_steps, n_delivery_starts, n_sims)
        mu_array : np.ndarray
            Drift array
        sigma_s, kappa_s, sigma_l, kappa_l : float
            Model parameters
        dw : np.ndarray
            Correlated shocks (n_sims, n_steps-1, 2)
        tau_array : np.ndarray
            Time-to-maturity array (n_steps, n_delivery_starts)

        Returns
        -------
        np.ndarray
            Updated forward array
        """
        n_steps, n_delivery_starts, n_sims = fwds_array.shape
        for m in range(n_delivery_starts):
            for t in range(1, n_steps):
                if tau_array[t, m] < 0:
                    break
                prev = fwds_array[t - 1, m, :]
                dw_s, dw_l = dw[:, t - 1, 0], dw[:, t - 1, 1]
                mu = mu_array[t, m]
                dF = prev * (mu * DT + sigma_s * np.exp(-kappa_s * tau_array[t, m]) * dw_s
                             + sigma_l * np.exp(-kappa_l * tau_array[t, m]) * dw_l)
                fwds_array[t, m, :] = prev + dF
        return fwds_array

    @cache_simulation
    def simulate(
        self,
        fwd_0: pd.Series,
        n_sims: int,
        **kwargs
    ) -> Tuple[xr.DataArray, xr.DataArray, pd.DataFrame]:
        """
        Simulate forward curves using internal simulation days.

        Parameters
        ----------
        fwd_0 : pd.Series
            Initial forward curve values indexed by delivery start dates.
        n_sims : int
            Number of simulation paths.
        **kwargs : dict
            Optional keyword arguments. Can be used to control caching, random seed, or other options.

        Returns
        -------
        Tuple[xr.DataArray, xr.DataArray, pd.DataFrame]
            - fwds: Simulated forward curves (xarray DataArray)
            - month_ahead: Month-ahead forwards derived from simulation (xarray DataArray)
            - spot_prices: Day-ahead spot prices (pandas DataFrame)
        """
        n_steps = len(self.simulation_days)
        dw = self.generate_dW(n_sims, n_steps) * np.sqrt(DT)

        delivery_start_dates = fwd_0.index
        n_delivery_starts = len(delivery_start_dates)

        # Initialize forward array
        fwds_array = np.zeros((n_steps, n_delivery_starts, n_sims))
        fwds_array[0, :, :] = np.tile(fwd_0.values[:, np.newaxis], (1, n_sims))

        # Precompute drift (mu) for all simulation days and maturities
        mu_df = pd.DataFrame(
            {maturity: -0.5 * self.log_var(maturity) for maturity in delivery_start_dates},
            index=self.simulation_days
        )
        mu_array = mu_df.values  # shape (n_steps, n_delivery_starts)

        # Precompute tau (time-to-maturity) array
        tau_array = np.array([[yfr(self.simulation_days[t], maturity)
                               for maturity in delivery_start_dates]
                              for t in range(n_steps)])

        # Accelerated simulation loop using Numba
        fwds_array = self._simulate_forward_loop(
            fwds_array, mu_array,
            self.sigma_s, self.kappa_s,
            self.sigma_l, self.kappa_l,
            dw, tau_array
        )

        # Convert to xarray DataArray
        coords = {
            SIMULATION_PATH: np.arange(n_sims),
            SIMULATION_DAY: self.simulation_days,
            DELIVERY_START: delivery_start_dates,
        }
        fwds = xr.DataArray(
            np.transpose(fwds_array, (2, 0, 1)),  # shape (n_sims, n_steps, n_delivery_starts)
            dims=[SIMULATION_PATH, SIMULATION_DAY, DELIVERY_START],
            coords=coords,
            name=FWD_CURVE,
        )

        # Extract month-ahead forwards
        month_ahead = extract_month_ahead(fwds)

        # Simulate day-ahead spot prices
        spot_prices = self._simulate_spot_prices(month_ahead, n_sims)

        return fwds, month_ahead, spot_prices


