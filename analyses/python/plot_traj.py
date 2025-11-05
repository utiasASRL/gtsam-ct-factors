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


def plotTrajectory(
    fname1,
    fname2=None,
    label1="GTSAM Trajectory",
    label2="STEAM Trajectory",
    fname_gt="../results/lost_gt.csv",
    step=None,
    plot_cov=False,
    plot_heading=True,
    plot_landmark_estimates=False,
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

    # If both fname1 and fname2 are given and plot_cov is True, plot side-by-side with covariances
    if fname2 is not None and plot_cov:
        df1, covariances1 = load_csv(fname1, step)
        df2, covariances2 = load_csv(fname2, step)
        df_gt, _ = load_csv(fname_gt, step)

        df1_lm = None
        df2_lm = None
        cov1_lm = None
        cov2_lm = None
        if plot_landmark_estimates:
            fname1_lm = fname1.replace(".csv", "_landmarks.csv")
            fname2_lm = fname2.replace(".csv", "_landmarks.csv")
            df1_lm, cov1_lm = load_csv(fname1_lm, None)
            df2_lm, cov2_lm = load_csv(fname2_lm, None)

        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
        for ax, df_plot, covs, title, color, df_lm, lm_cov in zip(
            axs,
            [df1, df2],
            [covariances1, covariances2],
            [label1, label2],
            ["r", "g"],
            [df1_lm, df2_lm],
            [cov1_lm, cov2_lm],
        ):
            x = df_plot["x"].to_numpy()
            y = df_plot["y"].to_numpy()
            theta = df_plot["theta"].to_numpy()
            x_gt = df_gt["x"].to_numpy()
            y_gt = df_gt["y"].to_numpy()
            theta_gt = df_gt["theta"].to_numpy()
            
            ax.plot(landmarks[:, 0], landmarks[:, 1], "ok", alpha=0.5, label="Landmarks")
            ax.plot(x, y, "-", label=title, alpha=0.9, color=color)
            ax.plot(x[0], y[0], "o", label="Start", color="k")
            ax.plot(x_gt, y_gt, "-", color="k", alpha=0.5, label="ground truth")

            if plot_landmark_estimates:
                lm_x = df_lm["x"].to_numpy()
                lm_y = df_lm["y"].to_numpy()
                ax.plot(lm_x, lm_y, "o", label="Estimated Landmarks", color=color)
                # Plot covariances of landmarks
                for lmx, lmy, lmcov in zip(lm_x, lm_y, lm_cov):
                    lmcov2d = lmcov[0:2, 0:2]
                    vals, vecs = np.linalg.eigh(lmcov2d)
                    angle = np.rad2deg(np.arctan2(*vecs[:, 1][::-1]))
                    factor = 3
                    height, width = factor * 2 * np.sqrt(vals)
                    ellipse = Ellipse(
                        (lmx, lmy),
                        width,
                        height,
                        angle=angle,
                        edgecolor=None,
                        alpha=0.2,
                        facecolor=color,
                        lw=1,
                    )
                    ax.add_patch(ellipse)
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
    if fname2 is not None and not plot_cov:
        df1, _ = load_csv(fname1, step)
        df2, _ = load_csv(fname2, step)
        df_gt, _ = load_csv(fname_gt, step)

        df1_lm = None
        df2_lm = None
        cov1_lm = None
        cov2_lm = None
        if plot_landmark_estimates:
            fname1_lm = fname1.replace(".csv", "_landmarks.csv")
            fname2_lm = fname2.replace(".csv", "_landmarks.csv")
            df1_lm, cov1_lm = load_csv(fname1_lm, None)
            df2_lm, cov2_lm = load_csv(fname2_lm, None)

        x1 = df1["x"].to_numpy()
        y1 = df1["y"].to_numpy()
        theta1 = df1["theta"].to_numpy()
        x2 = df2["x"].to_numpy()
        y2 = df2["y"].to_numpy()
        theta2 = df2["theta"].to_numpy()
        x_gt = df_gt["x"].to_numpy()
        y_gt = df_gt["y"].to_numpy()
        theta_gt = df_gt["theta"].to_numpy()

        plt.figure(figsize=(8, 6))
        plt.plot(landmarks[:, 0], landmarks[:, 1], "ok", alpha=0.5, label="Landmarks")
        plt.plot(x1, y1, "-", label=label1, alpha=0.9, color="r")
        plt.plot(x2, y2, "-", label=label2, alpha=0.9, color="g")
        plt.plot(x1[0], y1[0], "o", label="Start", color="k")
        plt.plot(x_gt, y_gt, "-", color="k", alpha=0.5, label="ground truth")
        if plot_landmark_estimates:
            lm1_x = df1_lm["x"].to_numpy()
            lm1_y = df1_lm["y"].to_numpy()
            lm2_x = df2_lm["x"].to_numpy()
            lm2_y = df2_lm["y"].to_numpy()
            plt.plot(lm1_x, lm1_y, "o", color="r")
            plt.plot(lm2_x, lm2_y, "o", color="g")
        if plot_heading:
            arrow_length = 0.2
            for xi, yi, ti in zip(x1, y1, theta1):
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
            for xi, yi, ti in zip(x2, y2, theta2):
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

    # If only fname1 is provided (fname2 is None), plot single trajectory
    df1, covariances1 = load_csv(fname1, step)
    df_gt, _ = load_csv(fname_gt, step)

    df1_lm = None
    cov1_lm = None
    if plot_landmark_estimates:
        fname1_lm = fname1.replace(".csv", "_landmarks.csv")
        df1_lm, cov1_lm = load_csv(fname1_lm, None)

    x1 = df1["x"].to_numpy()
    y1 = df1["y"].to_numpy()
    theta1 = df1["theta"].to_numpy()
    x_gt = df_gt["x"].to_numpy()
    y_gt = df_gt["y"].to_numpy()
    theta_gt = df_gt["theta"].to_numpy()

    plt.figure(figsize=(8, 6))
    plt.plot(landmarks[:, 0], landmarks[:, 1], "ok", alpha=0.5, label="Landmarks")
    plt.plot(x1, y1, "-", label=label1, alpha=0.9, color="r")
    plt.plot(x1[0], y1[0], "o", label="Start", color="k")
    plt.plot(x_gt, y_gt, "-", color="k", alpha=0.5, label="ground truth")
    if plot_landmark_estimates:
        lm_x = df1_lm["x"].to_numpy()
        lm_y = df1_lm["y"].to_numpy()
        plt.plot(lm_x, lm_y, "o", label="Estimated Landmarks", color="r")
        if plot_cov:
            for lmx, lmy, lmcov in zip(lm_x, lm_y, lm_cov):
                lmcov2d = lmcov[0:2, 0:2]
                vals, vecs = np.linalg.eigh(lmcov2d)
                angle = np.rad2deg(np.arctan2(*vecs[:, 1][::-1]))
                factor = 3
                height, width = factor * 2 * np.sqrt(vals)
                ellipse = Ellipse(
                    (lmx, lmy),
                    width,
                    height,
                    angle=angle,
                    edgecolor=None,
                    alpha=0.2,
                    facecolor=color,
                    lw=1,
                )
                ax.add_patch(ellipse)
    if plot_heading:
        arrow_length = 0.2
        for xi, yi, ti in zip(x1, y1, theta1):
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
        for xi, yi, ti, cov in zip(x1, y1, theta1, covariances1):
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
    fname_interp = "../results/lost_wrapper.csv"
    
    plotTrajectory(
        fname1=fname_interp,
        fname2=None,
        label1="Estimate",
        label2=None,
        plot_cov=True,
        step=1,
        plot_heading=False,
        plot_landmark_estimates=False,
    )
    
    