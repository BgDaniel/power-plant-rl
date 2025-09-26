import numpy as np
import matplotlib.pyplot as plt


def plot_observables(
    x,
    data,
    confidence_levels=(0.01, 0.05),
    path_index=None,
    ax=None,
    title=None,
    ylabel=None,
    show=True,
    figsize=(8, 4),
):
    """
    Plot a time series with mean, confidence intervals, and an optional single path overlay.

    Parameters
    ----------
    x : array-like
        The x-axis values (e.g., time or simulation days).
    data : pd.DataFrame
        2D data array where rows correspond to x-values and columns to different simulation paths.
    confidence_levels : tuple of float, default=(0.01, 0.05)
        Lower confidence levels for shading (e.g., 0.01 and 0.05 for 1% and 5%).
    path_index : int, optional
        Index of a single simulation path to overlay. If None, no path is overlaid.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, a new figure and axes are created.
    title : str, optional
        Title of the plot.
    ylabel : str, optional
        Label for the y-axis.
    show : bool, default=True
        Whether to call plt.show() at the end.
    figsize : tuple of float, default=(8, 4)
        Figure size if a new figure is created.

    Returns
    -------
    None
        The function plots the data on the given axes or creates a new figure.
    """
    lower1, lower2 = confidence_levels
    upper1, upper2 = 1 - lower1, 1 - lower2

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    mean_data = data.mean(axis=1)
    lower1_vals = np.percentile(data, lower1 * 100, axis=1)
    upper1_vals = np.percentile(data, upper1 * 100, axis=1)
    lower2_vals = np.percentile(data, lower2 * 100, axis=1)
    upper2_vals = np.percentile(data, upper2 * 100, axis=1)

    ax.plot(x, mean_data, lw=1.5, color="yellow", linestyle="--", label="Mean")
    ax.fill_between(
        x,
        lower2_vals,
        upper2_vals,
        color="blue",
        alpha=0.05,
        label=f"{int(lower2*100)}% CI",
    )
    ax.fill_between(
        x,
        lower1_vals,
        upper1_vals,
        color="blue",
        alpha=0.1,
        label=f"{int(lower1*100)}% CI",
    )

    if path_index is not None:
        ax.plot(
            x, data.iloc[:, path_index], color="red", lw=1, label=f"Path {path_index}"
        )

    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize="small", ncol=4)

    if show:
        plt.tight_layout()
        plt.show()
