import pandas as pd
from typing import Optional


class ForwardProduct:
    def __init__(
        self,
        name: str,
        delivery_start: pd.Timestamp,
        delivery_end: pd.Timestamp,
        prices: pd.Series,
        strike: Optional[float] = None,
        nominal: float = 1.0,
    ):
        """
        Represents a forward contract.

        Args:
            name: Name of the forward product.
            delivery_start: Start of the delivery period.
            delivery_end: End of the delivery period.
            prices: Forward prices indexed by date.
            strike: Strike price of the forward (default = first price).
            nominal: Contract size.
        """
        self.name: str = name
        self.start: pd.Timestamp = pd.Timestamp(delivery_start)
        self.end: pd.Timestamp = pd.Timestamp(delivery_end)
        self.prices: pd.Series = prices.sort_index()
        self.nominal: float = nominal
        self.strike: float = strike if strike is not None else prices.iloc[0]

    def covers(self, date: pd.Timestamp) -> bool:
        """
        Check if this forward covers a given date.

        Args:
            date: The date to check.

        Returns:
            True if the forward covers the date, False otherwise.
        """
        date = pd.Timestamp(date)
        return self.start <= date <= self.end

    def get_mark_to_market(self, current_price: float) -> float:
        """
        Compute mark-to-market value relative to the strike.

        Args:
            current_price: The current price of the forward.

        Returns:
            Nominal times the difference between current price and strike.
        """
        return self.nominal * (current_price - self.strike)