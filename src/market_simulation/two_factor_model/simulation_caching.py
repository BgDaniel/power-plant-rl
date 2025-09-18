import os
import hashlib
import json
import tempfile
import xarray as xr

import pandas as pd
from functools import wraps


# File name constants
FWDS_FILE = "fwds.nc"
MONTH_AHEAD_FILE = "month_ahead.csv"
SPOT_PRICES_FILE = "spot_prices.csv"

# Env var for cache dir
CACHE_FOLDER_ENV = "CACHE_FOLDER"


def cache_simulation(func):
    """
    Decorator that adds caching to the `simulate` method of TwoFactorForwardModel.
    Uses a hash of inputs and model parameters to create a unique cache folder.

    - Stores `fwds` (xarray.DataArray) as NetCDF.
    - Stores `month_ahead` and `spot_prices` (pandas DataFrames) as CSV.
    """
    @wraps(func)
    def wrapper(self, fwd_0: pd.Series, n_sims: int, *args, use_cache: bool = True, **kwargs):
        if not use_cache:
            return func(self, fwd_0, n_sims, *args, **kwargs)

        # Base cache folder
        base_cache = os.getenv(CACHE_FOLDER_ENV, tempfile.gettempdir())
        os.makedirs(base_cache, exist_ok=True)

        # Build hash key
        config_dict = {
            "as_of_date": str(getattr(self, "as_of_date", None)),
            "simulation_days": [str(d) for d in getattr(self, "simulation_days", [])],
            "params": getattr(self, "params", None).__dict__ if getattr(self, "params", None) else None,
            "config_path": getattr(self, "config_path", None),
            "sigma_s": getattr(self, "sigma_s", None),
            "kappa_s": getattr(self, "kappa_s", None),
            "sigma_l": getattr(self, "sigma_l", None),
            "kappa_l": getattr(self, "kappa_l", None),
            "rho": getattr(self, "rho", None),
            "beta": getattr(self, "beta", None),
            "sigma": getattr(self, "sigma", None),
            "n_sims": n_sims,
            "fwd_0": {"index": [str(d) for d in fwd_0.index], "values": fwd_0.values.tolist()},
        }
        json_str = json.dumps(config_dict, sort_keys=True)
        sim_hash = hashlib.md5(json_str.encode()).hexdigest()

        # Run-specific folder
        run_folder = os.path.join(base_cache, sim_hash)
        os.makedirs(run_folder, exist_ok=True)

        fwds_file = os.path.join(run_folder, FWDS_FILE)
        month_ahead_file = os.path.join(run_folder, MONTH_AHEAD_FILE)
        spot_prices_file = os.path.join(run_folder, SPOT_PRICES_FILE)

        # Try loading cache
        if os.path.exists(fwds_file) and os.path.exists(month_ahead_file) and os.path.exists(spot_prices_file):
            fwds = xr.load_dataarray(fwds_file)
            month_ahead = pd.read_csv(month_ahead_file, index_col=0, parse_dates=True)
            spot_prices = pd.read_csv(spot_prices_file, index_col=0, parse_dates=True)
            print(f"[CACHE] Loaded cached simulation from {run_folder}")
            return fwds, month_ahead, spot_prices

        # Run simulation
        fwds, month_ahead, spot_prices = func(self, fwd_0, n_sims, *args, **kwargs)

        # Save results
        fwds.to_netcdf(fwds_file)
        month_ahead.to_csv(month_ahead_file)
        spot_prices.to_csv(spot_prices_file)
        print(f"[CACHE] Saved simulation results to {run_folder}")

        return fwds, month_ahead, spot_prices

    return wrapper