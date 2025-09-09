import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Tuple

class MarketModel:
    def __init__(
        self,
        commodities: List[str],
        params_dict: Dict[str, Dict[str, float]],
        corr_matrix: np.ndarray,
        n_sims: int
    ) -> None:
        """
        Initialize the MarketModel.

        Args:
            commodities: List of commodity names.
            params_dict: Dictionary of parameters for each commodity.
            corr_matrix: Correlation matrix for the shocks.
            n_sims: Number of simulation paths per commodity.
        """
        self.commodities = commodities
        self.params_dict = params_dict
        self.corr_matrix = corr_matriximport numpy as np
        import pandas as pd
        import xarray as xr
        from typing import List, Dict, Callable, Tuple

        class MarketModel:
            def __init__(
                self,
                commodities: List[str],
                params_dict: Dict[str, Dict[str, float]],
                corr_matrix: np.ndarray,
                n_sims: int,
                n_delivery_months: int
            ) -> None:
                """
                Initialize the MarketModel.
        self.n_sims = n_sims
        self.chol = np.linalg.cholesky(corr_matrix)

    def simulate_paths(
        self,
        F0_dict: Dict[str, float],
        start_date: str,
        end_date: str
    ) -> Tuple[pd.DatetimeIndex, Dict[str, np.ndarray]]:
        """
        Simulate random paths for each commodity.

        Args:
            F0_dict: Initial forward price for each commodity.
            start_date: Simulation start date (YYYY-MM-DD).
            end_date: Simulation end date (YYYY-MM-DD).

        Returns:
            Tuple of date range and dictionary of simulated paths per commodity.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_steps = len(dates)
        results: Dict[str, np.ndarray] = {c: np.zeros((self.n_sims, n_steps)) for c in self.commodities}
        for c in self.commodities:
            results[c][:, 0] = F0_dict[c]
        for sim in range(self.n_sims):
            F_t: Dict[str, float] = {c: F0_dict[c] for c in self.commodities}
            for t in range(1, n_steps):
                dt = 1 / 252
                shocks = np.random.normal(size=2 * len(self.commodities))
                correlated_shocks = self.chol @ shocks
                for idx, c in enumerate(self.commodities):
                    p = self.params_dict[c]
                    mu: Callable[[float], float] = p['mu']
                    sigma_s: float = p['sigma_s']
                    kappa_s: float = p['kappa_s']
                    sigma_l: float = p['sigma_l']
                    kappa_l: float = p['kappa_l']
                    T = t * dt
                    dW1 = correlated_shocks[2 * idx]
                    dW2 = correlated_shocks[2 * idx + 1]
                    dF = F_t[c] * (
                        mu(T) * dt +
                        sigma_s * np.exp(-kappa_s * T) * dW1 +
                        sigma_l * np.exp(-kappa_l * T) * dW2
                    )
                    F_t[c] += dF
                    results[c][sim, t] = F_t[c]
        return dates, results