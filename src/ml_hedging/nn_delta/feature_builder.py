import numpy as np

class FeatureBuilder:
    """
    Builds features for the neural network (NN) delta model.

    Parameters
    ----------
    efficiency : float
        Efficiency factor of the power plant used to compute the spread:
        spread = fwd_power - efficiency * fwd_coal.
    """

    def __init__(self, efficiency: float):
        self.efficiency: float = efficiency

    def build(
        self,
        fwd_power: np.ndarray,
        fwd_coal: np.ndarray,
        beta_power: np.ndarray,
        beta_coal: np.ndarray,
        t: float,
        sigma_power: float | None = None,
        sigma_coal: float | None = None,
        corr_power_coal: float | None = None,
    ) -> np.ndarray:
        """
        Build the feature matrix for NN input.

        Parameters
        ----------
        fwd_power : np.ndarray
            Forward prices of power (shape: n_samples,).
        fwd_coal : np.ndarray
            Forward prices of coal (shape: n_samples,).
        beta_power : np.ndarray
            Beta exposures for power (shape: n_samples,).
        beta_coal : np.ndarray
            Beta exposures for coal (shape: n_samples,).
        t : float
            Delivery month start as float (can be ordinal date).
        sigma_power : float, optional
            Volatility of power, used if you want to include it as feature.
        sigma_coal : float, optional
            Volatility of coal, used if you want to include it as feature.
        corr_power_coal : float, optional
            Correlation between power and coal, used if included as feature.

        Returns
        -------
        np.ndarray
            Feature matrix of shape (n_samples, n_features).
            Features are stacked as:
            [fwd_power, fwd_coal, spread, beta_power, beta_coal, t,
             sigma_power?, sigma_coal?, corr_power_coal?]
        """
        # Compute spread
        spread = fwd_power - self.efficiency * fwd_coal

        # Base features
        features = [fwd_power, fwd_coal, spread, beta_power, beta_coal, np.full_like(fwd_power, t)]

        # Optional vol info
        if sigma_power is not None:
            features.append(np.full_like(fwd_power, sigma_power))
        if sigma_coal is not None:
            features.append(np.full_like(fwd_coal, sigma_coal))
        if corr_power_coal is not None:
            features.append(np.full_like(fwd_power, corr_power_coal))

        # Stack into a matrix: shape (n_samples, n_features)
        X = np.column_stack(features)
        return X


if __name__ == "__main__":
    # Simple sanity check
    n_samples = 5
    fwd_power = np.linspace(50, 150, n_samples)
    fwd_coal = np.linspace(70, 90, n_samples)
    beta_power = np.ones(n_samples) * 0.8
    beta_coal = np.ones(n_samples) * 1.2
    t = 738000  # example ordinal date

    builder = FeatureBuilder(efficiency=0.4)
    X = builder.build(fwd_power, fwd_coal, beta_power, beta_coal, t, sigma_power=0.1, sigma_coal=0.05, corr_power_coal=0.2)

    print("Feature matrix shape:", X.shape)
    print("Sample features:\n", X)
