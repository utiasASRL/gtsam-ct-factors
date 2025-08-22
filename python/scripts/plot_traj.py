import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
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
        cov = np.array(
            [
                [row["C11"], row["C12"], row["C13"]],
                [row["C12"], row["C22"], row["C23"]],
                [row["C13"], row["C23"], row["C33"]],
            ]
        )
        covariances.append(cov)

    return df, covariances


def plotTrajectory(
    fname,
    fname_gt="/home/cho/gtsam-ct-factors/results/lost_gt.csv",
    fname_steam=None,
    step=None,
    plot_cov=False,
    plot_heading=True,
):
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

    # If both fname and fname_steam are given and plot_cov is True, plot side-by-side with covariances
    if fname_steam is not None and plot_cov:
        df, covariances = load_csv(fname, step)
        df_steam, covariances_steam = load_csv(fname_steam, step)
        df_gt, _ = load_csv(fname_gt, step)

        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
        for ax, df_plot, covs, title, color in zip(
            axs,
            [df, df_steam],
            [covariances, covariances_steam],
            ["GTSAM Trajectory", "STEAM Trajectory"],
            ["r", "g"],
        ):
            x = df_plot["x"].to_numpy()
            y = df_plot["y"].to_numpy()
            theta = df_plot["theta"].to_numpy()
            ax.plot(landmarks[:, 0], landmarks[:, 1], "og", label="Landmarks")
            ax.plot(x, y, "-", label=title, alpha=0.9, color=color)
            ax.plot(x[0], y[0], "o", label="Start", color="k")
            ax.plot(df_gt["x"], df_gt["y"], "-", color="k", alpha=0.5, label="ground truth")
            # Plot heading arrows if requested
            if plot_heading:
                arrow_length = 0.2
                for xi, yi, ti in zip(x, y, theta):
                    ax.arrow(
                        xi,
                        yi,
                        arrow_length * np.cos(ti),
                        arrow_length * np.sin(ti),
                        head_width=0.05,
                        head_length=0.1,
                        fc=color,
                        ec=color,
                        alpha=0.5,
                    )
            # Plot covariance ellipses
            for xi, yi, ti, cov in zip(x, y, theta, covs):
                cov2d = cov[0:2, 0:2]
                vals, vecs = np.linalg.eigh(cov2d)
                angle = np.rad2deg(np.arctan2(*vecs[:, 1][::-1]))
                factor = 3
                height, width = factor * 2 * np.sqrt(vals)
                ellipse = Ellipse(
                    (xi, yi),
                    width,
                    height,
                    angle=angle,
                    edgecolor=None,
                    alpha=0.2,
                    facecolor=color,
                    lw=1,
                )
                ax.add_patch(ellipse)
            ax.set_aspect("equal", "box")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title(title)
            ax.legend()
        plt.tight_layout()
        plt.show()
        return

    # If plot_cov is False, plot both trajectories on the same plot
    if fname_steam is not None and not plot_cov:
        df, _ = load_csv(fname, step)
        df_steam, _ = load_csv(fname_steam, step)
        df_gt, _ = load_csv(fname_gt, step)

        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        theta = df["theta"].to_numpy()
        x_steam = df_steam["x"].to_numpy()
        y_steam = df_steam["y"].to_numpy()
        theta_steam = df_steam["theta"].to_numpy()

        plt.figure(figsize=(8, 6))
        plt.plot(landmarks[:, 0], landmarks[:, 1], "og", label="Landmarks")
        plt.plot(x, y, "-", label="Traj - GTSAM", alpha=0.9, color="r")
        plt.plot(x_steam, y_steam, "-", label="Traj - STEAM", alpha=0.9, color="g")
        plt.plot(x[0], y[0], "o", label="Start", color="k")
        plt.plot(df_gt["x"], df_gt["y"], "-", color="k", alpha=0.5, label="ground truth")
        if plot_heading:
            arrow_length = 0.2
            for xi, yi, ti in zip(x, y, theta):
                plt.arrow(
                    xi,
                    yi,
                    arrow_length * np.cos(ti),
                    arrow_length * np.sin(ti),
                    head_width=0.05,
                    head_length=0.1,
                    fc="r",
                    ec="r",
                    alpha=0.5,
                )
            for xi, yi, ti in zip(x_steam, y_steam, theta_steam):
                plt.arrow(
                    xi,
                    yi,
                    arrow_length * np.cos(ti),
                    arrow_length * np.sin(ti),
                    head_width=0.05,
                    head_length=0.1,
                    fc="g",
                    ec="g",
                    alpha=0.5,
                )
        plt.gca().set_aspect("equal", "box")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Trajectories")
        plt.axis("equal")
        plt.legend()
        plt.show()
        return

    # If only fname is provided (fname_steam is None), plot single trajectory
    df, covariances = load_csv(fname, step)
    df_gt, _ = load_csv(fname_gt, step)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    theta = df["theta"].to_numpy()

    plt.figure(figsize=(8, 6))
    plt.plot(landmarks[:, 0], landmarks[:, 1], "og", label="Landmarks")
    plt.plot(x, y, "-", label="Trajectory", alpha=0.9, color="r")
    plt.plot(x[0], y[0], "o", label="Start", color="k")
    plt.plot(df_gt["x"], df_gt["y"], "-", color="k", alpha=0.5, label="ground truth")
    if plot_heading:
        arrow_length = 0.2
        for xi, yi, ti in zip(x, y, theta):
            plt.arrow(
                xi,
                yi,
                arrow_length * np.cos(ti),
                arrow_length * np.sin(ti),
                head_width=0.05,
                head_length=0.1,
                fc="r",
                ec="r",
                alpha=0.5,
            )
    if plot_cov:
        for xi, yi, ti, cov in zip(x, y, theta, covariances):
            cov2d = cov[0:2, 0:2]
            vals, vecs = np.linalg.eigh(cov2d)
            angle = np.rad2deg(np.arctan2(*vecs[:, 1][::-1]))
            factor = 3
            height, width = factor * 2 * np.sqrt(vals)
            ellipse = Ellipse(
                (xi, yi),
                width,
                height,
                angle=angle,
                edgecolor=None,
                alpha=0.2,
                facecolor="b",
                lw=1,
            )
            plt.gca().add_patch(ellipse)
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Trajectory")
    plt.axis("equal")
    plt.legend()
    plt.show()
    return



if __name__ == "__main__":
    fname = "/home/cho/gtsam-ct-factors/results/lost.csv"
    fname_steam = "/home/cho/gtsam-ct-factors/results/lost_steam.csv"
    fname_interp = "/home/cho/gtsam-ct-factors/results/lost_interp.csv"
    # plotTrajectory(
    #     fname=fname,
    #     fname_steam=fname_steam,
    #     plot_cov=False,
    #     step=2,
    #     plot_heading=False,
    # )
    plotTrajectory(
        fname=fname,
        fname_steam=fname_interp,
        plot_cov=True,
        step=1,
        plot_heading=False,
    )
    
    