import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
import sys
import matplotlib.pyplot as plt


def load_csv(fname, step=None):
    df = pd.read_csv(fname)
    # subsample rows
    if step:
        nrows = len(df)
        keeprows = range(0, len(df), step)
        df = df.iloc[keeprows]

    # Extract covariance elements into a list
    covariances = []
    for _, row in df.iterrows():
        # Get covariance
        if "C33" in row: # Detect size of covariance matrix (pose or landmark)
            cov = np.array(
            [
                [row["C11"], row["C12"], row["C13"]],
                [row["C12"], row["C22"], row["C23"]],
                [row["C13"], row["C23"], row["C33"]],
            ]
            )
        else:
            cov = np.array(
            [
                [row["C11"], row["C12"]],
                [row["C12"], row["C22"]],
            ]
        )
        covariances.append(cov)

    return df, covariances

# Use load_csv in plot_se2_trajectory
def plot_se2_trajectory(ax, csv_path, title):
    df, covariances = load_csv(csv_path)
    x = df['x']
    y = df['y']
    theta = df['theta']

    ax.plot(x, y, marker='o', linestyle='-', color='tab:blue', label='Trajectory')
    ax.quiver(x, y, np.cos(theta), np.sin(theta), angles='xy', scale_units='xy', scale=5, width=0.005, color='tab:orange', label='Heading')

    # Plot covariance ellipses if available and 2D
    for xi, yi, cov in zip(x, y, covariances):
        plot_cov_ellipse(ax, xi, yi, cov, n_std=1, edgecolor='tab:green', facecolor='tab:green', lw=1, alpha=0.25)

    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
                

def plot_cov_ellipse(ax, xi, yi, cov, n_std=1.0, **kwargs):
    """Plot a covariance ellipse centered at mean with covariance cov."""
    cov2d = cov[0:2, 0:2]
    vals, vecs = np.linalg.eigh(cov2d)
    angle = np.rad2deg(np.arctan2(*vecs[:, 1][::-1]))
    height, width = n_std * 2 * np.sqrt(vals)
    ellipse = Ellipse(
        (xi, yi),
        width,
        height,
        angle=angle,
        **kwargs
    )
    ax.add_patch(ellipse)


if __name__ == "__main__":
    paths = ["results/simple_ex_full.csv",
             "results/simple_ex_interp.csv",
             "results/simple_ex_recov.csv"]
    titles = ["full solve", 
              "interpolated",
              "interp + recover mean"]
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for i, csv_path in enumerate(paths):
        plot_se2_trajectory(axs[i], csv_path, titles[i])
    plt.tight_layout()
    plt.show()