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

def plot_axes_at_points(ax, x, y, theta, length=0.2, greyscale=False):
    """Plot small 2D frames (axes) at each (x, y, theta) pose."""
    for xi, yi, thetai in zip(x, y, theta):
        # Origin
        # color_o = 'k' if not greyscale else (0.3, 0.3, 0.3, 0.7)
        color_x = 'r' if not greyscale else (0.5, 0.5, 0.5, 0.7)
        color_y = 'g' if not greyscale else (0.7, 0.7, 0.7, 0.7)
        # ax.plot(xi, yi, marker='o', color=color_o, markersize=2)
        ax.arrow(xi, yi, length * np.cos(thetai), length * np.sin(thetai),
                    head_width=0.05, head_length=0.05, fc=color_x, ec=color_x, linewidth=1)
        ax.arrow(xi, yi, -length * np.sin(thetai), length * np.cos(thetai),
                    head_width=0.05, head_length=0.05, fc=color_y, ec=color_y, linewidth=1)

def plot_se2_trajectory(ax, csv_path, title, greyscale=False):
    df, covariances = load_csv(csv_path)
    x = df['x']
    y = df['y']
    theta = df['theta']

    plot_axes_at_points(ax, x, y, theta, length=0.2, greyscale=greyscale)

    # Plot covariance ellipses if available and 2D
    for xi, yi,ti, cov in zip(x, y,theta, covariances):
        if greyscale:
            plot_cov_ellipse(ax, xi, yi,ti, cov, n_std=1, edgecolor=(0.5, 0.5, 0.5, 0.25), facecolor=(0.8, 0.8, 0.8, 0.25), lw=1, alpha=0.25)
        else:
            plot_cov_ellipse(ax, xi, yi,ti, cov, n_std=1, edgecolor=None, facecolor='tab:blue', lw=1, alpha=0.25)

    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title)
    # ax.grid(True)

def plot_cov_ellipse(ax, xi, yi, ti, cov, n_std=1.0, **kwargs):
    """Plot a covariance ellipse centered at mean with covariance cov."""
    cov2d = cov[0:2, 0:2]
    vals, vecs = np.linalg.eigh(cov2d)
    angle = np.rad2deg(np.arctan2(*vecs[:, 1][::-1]) + ti)
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
    paths = ["analyses/results/simple_ex_full.csv",
             "analyses/results/simple_ex_estim.csv",
             "analyses/results/simple_ex_interp.csv",
             "analyses/results/simple_ex_interp_graph.csv"]
    titles = ["full solve", 
              "just estimated",
              "interpolated",
              "interp via graph"]
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    
    axs = axs.flatten()
    for i, csv_path in enumerate(paths):
        if i == 0:
            plot_se2_trajectory(axs[i], csv_path, titles[i])
        else:
            plot_se2_trajectory(axs[i], paths[0], titles[i], greyscale=True)
            plot_se2_trajectory(axs[i], csv_path, titles[i])
    # Link all axes in x and y
    for ax in axs.flatten()[1:]:
        ax.sharex(axs.flatten()[0])
        ax.sharey(axs.flatten()[0])
    plt.tight_layout()
    plt.show()