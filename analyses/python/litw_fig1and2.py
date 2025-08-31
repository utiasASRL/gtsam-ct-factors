import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

"""
Plots figures for Figure 2 of the Lost in the woods section.
Ground truth trajectory and a localization using odometry 
and bearing range measurements.
"""

# GT landmark locations
landmarks = np.array(
    [
        [5.3648, 0.6713],
        [5.6713, -0.9840],
        [5.4763, -2.3006],
        [5.4534, 2.8198],
        [7.2665, 1.7578],
        [7.8653, -0.5483],
        [7.2634, -1.9018],
        [9.5005, -1.8431],
        [8.9889, 0.7254],
        [3.5591, -1.1357],
        [0.7656, -1.9158],
        [-1.2675, -1.4736],
        [-1.2095, 0.9508],
        [0.5486, 0.0814],
        [-0.2035, 2.5533],
        [2.8177, 2.7351],
        [3.3851, 1.3623],
    ]
)

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

def plot_axes_at_points(ax, x, y, theta, length=0.2, downsample=1, gt_vals=False):
    """Plot small 2D frames (axes) at each (x, y, theta) pose."""
        
    color  = 'b' if not gt_vals else 'g'
    alpha = 1.0 if not gt_vals else 0.7
    length_factor = 1.0
    length = length_factor * length
    head_width = 0.05 * length_factor
    head_length = 0.05* length_factor
    
    # plot axes
    for xi, yi, thetai in zip(x, y, theta):
        ax.arrow(xi, yi, length * np.cos(thetai), length * np.sin(thetai),
                    head_width=head_width, head_length=head_length, fc=color, ec=color,alpha=alpha,  linewidth=1)
        ax.arrow(xi, yi, -length * np.sin(thetai), length * np.cos(thetai),
                    head_width=head_width, head_length=head_length, fc=color, ec=color, alpha=alpha, linewidth=1)

def plot_se2_trajectory(ax, csv_path, title, plot_cov=False, gt_vals=False, downsample=5):
    df, covariances = load_csv(csv_path)
    x = df['x'].values
    y = df['y'].values
    theta = df['theta'].values
    
    # Plot trajectory
    color = 'b' if not gt_vals else 'g'
    alpha = 0.3
    ax.plot(x, y, '-', color=color, alpha=alpha, linewidth=0.7)
    
    # Downsample points 
    x_ds = x[::downsample]
    y_ds = y[::downsample]
    th_ds = theta[::downsample]
    cov_ds = covariances[::downsample]
    # Plot axes
    plot_axes_at_points(ax, x_ds, y_ds, th_ds, length=0.2,downsample=1, gt_vals=gt_vals)
    
    
    # Plot covariance ellipses if available and 2D
    if plot_cov:
        for xi, yi, ti, cov in zip(x_ds, y_ds, th_ds, cov_ds):
            if gt_vals:
                plot_cov_ellipse(ax, xi, yi, ti, cov, n_std=3, edgecolor=None, facecolor='g', lw=1, alpha=0.25)
            else:
                plot_cov_ellipse(ax, xi, yi, ti, cov, n_std=3, edgecolor=None, facecolor='b', lw=1, alpha=0.25)

    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title)
    # ax.grid(True)

def plot_se2_points(ax, csv_path, title, downsample=5):
    df, _ = load_csv(csv_path)
    x = df['x'].values[::downsample]
    y = df['y'].values[::downsample]

    ax.plot(x,y,'-b', linewidth=0.5)
    
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title)


def plot_start_end(ax, csv_path):
    df, _ = load_csv(csv_path)
    x = df['x'].values
    y = df['y'].values

    ax.plot(x[-1],y[-1],'*r')
    ax.plot(x[0],y[0],'*g')
    
def plot_landmarks(ax):
    ax.plot(landmarks[:,0], landmarks[:,1], 'og')

def plot_cov_ellipse(ax, xi, yi, ti, cov, n_std=3.0, **kwargs):
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
    path_gt_full = "analyses/results/lost_gt_full.csv"
    path_gt = "analyses/results/litw_fig2_gt.csv"
    path_solve = "analyses/results/litw_fig2_gtsam.csv"
    titles = ["Ground Truth Trajectory"
              "GTSAM Solve"]
    downsample=40


    # Plot Ground Truth
    fig1,ax =plt.subplots(1,1,figsize=(8,4))
    plot_landmarks(ax)
    plot_se2_points(ax, path_gt_full, "Ground Truth Trajectory", downsample=1)
    plot_start_end(ax, path_gt_full)
    # Plot GTSAM Solution
    fig2,ax =plt.subplots(1,1,figsize=(8,4))
    plot_landmarks(ax)
    plot_se2_trajectory(ax, path_gt, None, gt_vals=True, downsample=downsample)
    plot_se2_trajectory(ax, path_solve, "GTSAM Solve", plot_cov=True, downsample=downsample)
    
    plt.show(block=False)
    dpi=800
    save_figs = input("Save figures? [Y/n]: ").strip().lower()
    if save_figs == "" or save_figs == "y":
        fig1.savefig("analyses/plots/litw_fig1_gt.png", dpi=dpi, bbox_inches='tight')
        fig2.savefig("analyses/plots/litw_fig2_solve.png", dpi=dpi, bbox_inches='tight')
    