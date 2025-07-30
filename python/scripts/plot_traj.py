import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt


def plotTrajectory(filename, step=None, plot_cov=False, plot_heading=True):
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

    df = pd.read_csv(filename)
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

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    theta = df["theta"].to_numpy()

    plt.figure(figsize=(8, 6))
    # plot landmarks
    plt.plot(landmarks[:, 0], landmarks[:, 1], "og", label=("Landmarks"))
    # trajectory
    plt.plot(x, y, "-", label="Position", alpha=0.9, color="b")
    # Starting point
    plt.plot(x[0], y[0], "o", label="Start", color="k")
    # Plot heading arrows
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
    # Plot covariance ellipses
    if plot_cov:
        for xi, yi, ti, cov in zip(x, y, theta, covariances):
            cov2d = cov[:2, :2]
            vals, vecs = np.linalg.eigh(cov2d)
            angle = np.rad2deg(np.arctan2(*vecs[:, 1][::-1]) + ti)
            factor = 5  # np.sqrt(2.447746830681) # 95 % samples inside ellipse
            height, width = factor * 2 * np.sqrt(vals)  # factor of 2 for diameter
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


if __name__ == "__main__":
    filename = "../../results/lost.csv"
    plotTrajectory(filename=filename, plot_cov=True, step=1, plot_heading=False)
