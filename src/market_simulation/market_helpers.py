import pandas as pd
import numpy as np
import xarray as xr
import datetime as dt
from pandas.tseries.offsets import MonthBegin

from src.market_simulation.constants import (
    SIMULATION_DAY,
    SIMULATION_PATH,
    DELIVERY_START,
)


def extract_month_ahead(fwds: xr.DataArray) -> pd.DataFrame:
    """
    Extract the month-ahead forward curve from absolute forward simulations.

    Parameters
    ----------
    fwds : xr.DataArray
        Forward simulations with dims [simulation, day, delivery_start].

    Returns
    -------
    pd.DataFrame
        Month-ahead forward prices:
        - Index: days
        - Columns: simulations
    """
    simulation_days = fwds.coords[SIMULATION_DAY].to_index()
    simulation_paths = fwds.coords[SIMULATION_PATH].values

    # map each day -> next delivery start
    delivery_start_dates = pd.DatetimeIndex(
        [pd.Timestamp(day) + pd.offsets.MonthBegin(1) for day in simulation_days]
    )

    # container for results
    data = np.zeros((len(simulation_days), len(simulation_paths)))

    for i, (simulation_day, delivery_start_date) in enumerate(
        zip(simulation_days, delivery_start_dates)
    ):
        if delivery_start_date not in delivery_start_dates:
            raise ValueError(
                f"Delivery start {delivery_start_date} not in fwds.delivery_start coords"
            )

        # pick values for that day and cap across all sims
        try:
            data[i, :] = fwds.sel(
                SIMULATION_DAY=simulation_day, DELIVERY_START=delivery_start_date
            ).values
        except KeyError:
            data[i, :] = np.nan  # fill with NaN if indices not found

    return pd.DataFrame(data, index=simulation_days, columns=simulation_paths)

def yfr(start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    """
    Compute year fraction between two dates using simple day count / 365.

    Parameters
    ----------
    start_date : pd.Timestamp
        Starting date.
    end_date : pd.Timestamp
        Ending date.

    Returns
    -------
    float
        Year fraction between the two dates.
    """
    return (end_date - start_date).days / 365.0
