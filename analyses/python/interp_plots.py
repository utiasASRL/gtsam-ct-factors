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

def plot_axes_at_points(ax, x, y, theta, length=0.2, interp=False, full_slv=False):
    """Plot small 2D frames (axes) at each (x, y, theta) pose."""
        
    color  = 'b' if not full_slv else 'g'
    alpha = 1.0 if not (full_slv or interp) else 0.2
    length_factor = 1.0 if not (full_slv or interp) else 0.5
    length = length_factor * length
    head_width = 0.05 * length_factor
    head_length = 0.05* length_factor
    
    # plot axes
    for xi, yi, thetai in zip(x, y, theta):
        ax.arrow(xi, yi, length * np.cos(thetai), length * np.sin(thetai),
                    head_width=head_width, head_length=head_length, fc=color, ec=color,alpha=alpha,  linewidth=1)
        ax.arrow(xi, yi, -length * np.sin(thetai), length * np.cos(thetai),
                    head_width=head_width, head_length=head_length, fc=color, ec=color, alpha=alpha, linewidth=1)

def plot_se2_trajectory(ax, csv_path, title=None, interp=False, plot_cov=False, full_slv=False, downsample=1):
    df, covariances = load_csv(csv_path)
    x = df['x'].values
    y = df['y'].values
    theta = df['theta'].values
    
    # Plot trajectory
    color = 'b' if not full_slv else 'g'
    alpha = 0.7
    if interp or full_slv:
        ax.plot(x, y, '-', color=color, alpha=alpha, linewidth=1.0)
    
    # Downsample points 
    x_ds = x[::downsample]
    y_ds = y[::downsample]
    th_ds = theta[::downsample]
    cov_ds = covariances[::downsample]
    # Plot axes
    plot_axes_at_points(ax, x_ds, y_ds, th_ds, length=0.2,interp=interp, full_slv=full_slv)
    
    # Plot covariance ellipses if available and 2D
    if plot_cov:
        color = 'b' if not full_slv else 'g'
        for xi, yi, ti, cov in zip(x_ds, y_ds, th_ds, cov_ds):
            if interp or full_slv:
                plot_cov_ellipse(ax, xi, yi, ti, cov, n_std=3, edgecolor=color, facecolor=color, lw=1, alpha=0.1)
            else:
                plot_cov_ellipse(ax, xi, yi, ti, cov, n_std=3, edgecolor=color, facecolor=color, lw=1, alpha=0.25)

    ax.set_aspect('equal')
    # ax.set_xlabel('x [m]')
    # ax.set_ylabel('y [m]')
    ax.set_title(title)
    # ax.grid(True)

def plot_se2_points(ax, csv_path, title=None, downsample=5):
    df, _ = load_csv(csv_path)
    x = df['x'].values[::downsample]
    y = df['y'].values[::downsample]

    ax.plot(x,y,'-b', linewidth=0.5)
    
    ax.set_aspect('equal')
    # ax.set_xlabel('x [m]')
    # ax.set_ylabel('y [m]')
    ax.set_title(title)
    

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
    path_full = "analyses/results/simple_ex_full.csv"
    path_est = "analyses/results/simple_ex_estim.csv"
    path_interp = "analyses/results/simple_ex_interp.csv"
    path_graph_interp = "analyses/results/simple_ex_interp_graph.csv"
   
    # Plot Interpolated Solve
    def plot_fig(title_on=True):
        fig,ax =plt.subplots(3,1,figsize=(8,6),sharex=True, sharey=True)
        plot_se2_trajectory(ax[0], path_full, full_slv=True, plot_cov=True, downsample=2)
        plot_se2_trajectory(ax[0], path_est, plot_cov=True, downsample=1)
        if title_on:
            ax[0].set_title('Main Solve - Full vs Reduced',loc='left')
        ax[0].get_xaxis().set_ticks([])
        ax[0].get_yaxis().set_ticks([])
        
        plot_se2_trajectory(ax[1], path_full, full_slv=True,plot_cov=True, downsample=2)
        plot_se2_trajectory(ax[1], path_interp, plot_cov=True, interp=True, downsample=2)
        if title_on:
            ax[1].set_title('Interpolated - Covariance interpolated via equation:',loc='left')
        ax[1].get_xaxis().set_ticks([])
        ax[1].get_yaxis().set_ticks([])
        
        plot_se2_trajectory(ax[2], path_full, full_slv=True,plot_cov=True,downsample=2)
        plot_se2_trajectory(ax[2], path_graph_interp, plot_cov=True,interp=True, downsample=2)
        if title_on:
            ax[2].set_title('Interpolated - Covariance interpolated via graph:',loc='left')
        ax[2].get_xaxis().set_ticks([])
        ax[2].get_yaxis().set_ticks([])
        return fig, ax
    
    dpi=800
    fig, ax = plot_fig()
    plt.show(block=False)
    save_figs = input("Save figure? [Y/n]: ").strip().lower()
    if save_figs == "" or save_figs == "y":
        fig.savefig("analyses/plots/interp_fig1.png", dpi=dpi, bbox_inches='tight')
        # Plot again with zoom
        fig, ax = plot_fig(False)
        plt.show(block=False)
        save_figs = input("Save Zoomed figure? [Y/n]: ").strip().lower()
        if save_figs == "" or save_figs == "y":
            fig.savefig("analyses/plots/interp_fig1_zoom.png", dpi=dpi, bbox_inches='tight')
            
            
    
