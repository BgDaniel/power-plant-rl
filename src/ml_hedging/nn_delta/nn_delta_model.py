import torch
import torch.nn as nn

class NNDeltaModel(nn.Module):
    """
    Simple feed-forward neural network to predict hedge delta positions
    for power and coal.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dim : int, default=64
        Number of units in each hidden layer.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # outputs: delta_power, delta_coal
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (n_samples, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (n_samples, 2), containing
            delta_power and delta_coal.
        """
        return self.net(x)


if __name__ == "__main__":
    # ----------------------------
    # Quick sanity test
    # ----------------------------
    import numpy as np

    n_samples = 5
    n_features = 6

    # Random input
    X = torch.tensor(np.random.rand(n_samples, n_features), dtype=torch.float32)

    # Initialize model
    model = NNDeltaModel(input_dim=n_features, hidden_dim=16)

    # Forward pass
    out = model(X)
    print("Input shape:", X.shape)
    print("Output shape:", out.shape)
    print("Sample output:\n", out.detach().numpy())
