from dataclasses import dataclass
from src.base_config import BaseConfig


@dataclass(frozen=True)
class TwoFactorModelConfig(BaseConfig):
    """
    Unified configuration for both the two-factor forward model
    and the day-ahead OU process.

    Attributes
    ----------
    sigma_s : float
        Volatility of the short-term factor (two-factor model).
    kappa_s : float
        Mean reversion speed of the short-term factor (two-factor model).
    sigma_l : float
        Volatility of the long-term factor (two-factor model).
    kappa_l : float
        Mean reversion speed of the long-term factor (two-factor model).
    rho : float
        Correlation between short- and long-term shocks (two-factor model).

    beta : float
        Volatility parameter of the day-ahead OU process.
    kappa : float
        Mean reversion speed of the day-ahead OU process.
    """

    sigma_s: float
    kappa_s: float
    sigma_l: float
    kappa_l: float
    rho: float

    beta: float
    kappa: float
