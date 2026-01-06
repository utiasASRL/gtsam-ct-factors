import numpy as np
import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt

csv_path_fixed_loc = Path("../results/woods_timing_results/fixed_noise_loc.csv")
csv_path_fixed_slam = Path("../results/woods_timing_results/fixed_noise_slam.csv")
csv_path_non_fixed_loc = Path("../results/woods_timing_results/non_fixed_noise_loc.csv")
csv_path_non_fixed_slam = Path("../results/woods_timing_results/non_fixed_noise_slam.csv")

# Figure 1 - Localization and fixed noise

df = pd.read_csv(csv_path_fixed_loc)
print(f"Loaded {len(df)} rows from {csv_path_fixed_loc}")

interp_period = df.iloc[:, 0].to_numpy()

solve_time_full = df.iloc[:, 1].to_numpy()*1e-6 # convert to s
solve_time_full_iter = df.iloc[:, 3].to_numpy()*1e-6 # convert to s

#Take average of full solve time for clearer visualization
avg_solve_time_full = solve_time_full.mean()
solve_time_full = np.full_like(solve_time_full, avg_solve_time_full)

avg_solve_time_full_iter = solve_time_full_iter.mean()
solve_time_full_iter = np.full_like(solve_time_full_iter, avg_solve_time_full_iter)


acc_full = df.iloc[:, 4].to_numpy()
nees_full = df.iloc[:, 5].to_numpy()

solve_time_inter = df.iloc[:, 6].to_numpy()*1e-6 # convert to s
solve_time_inter_iter = df.iloc[:, 8].to_numpy()*1e-6 # convert to s
acc_inter_all = df.iloc[:, 9].to_numpy()
acc_inter_est = df.iloc[:, 10].to_numpy()
nees_inter_all = df.iloc[:, 11].to_numpy()
nees_inter_est = df.iloc[:, 12].to_numpy()


plt.figure(figsize=(7, 6))
plt.xlabel("Interval of Non-Interpolated States")

#Name of plot
plt.title("Localization w/ Simplified Noise Model")

#Plot full vs interpolated solve time
plt.plot(interp_period, solve_time_full, label="Full Graph Solve Time", linestyle='--', marker=None, color='blue')
plt.plot(interp_period, solve_time_inter, label="Inter. Graph Solve Time", marker='o', color='blue')

#Plot full vs interpolated solve time per iteration
plt.plot(interp_period, solve_time_full_iter, label="Full Graph Iter. Time", linestyle='--', marker=None, color='lightblue')
plt.plot(interp_period, solve_time_inter_iter, label="Inter. Graph Iter. Time", marker='o', color='lightblue')

# put accuracy on same plot with different y-axis
plt.twinx()
plt.plot(interp_period, acc_full, label="Full Graph Acc.", linestyle='--', marker=None, color='green')
plt.plot(interp_period, acc_inter_all, label="Inter. Graph Acc. (All Times)", marker='o', color='green')
plt.plot(interp_period, acc_inter_est, label="Inter. Graph Acc. (Est. Times)", marker='o', color='lightgreen')

ax_right = plt.gca()
ax_right.set_ylim(0, 0.2)
ax_right.set_ylabel("Accuracy [m]")

ax_left = plt.gcf().axes[0]
ax_left.set_ylim(0, 0.25)
ax_left.set_ylabel("Time [s]")


#color code axis labels, ticks and lines to match plots
ax_left.yaxis.label.set_color('blue')
ax_right.yaxis.label.set_color('green')
ax_left.tick_params(axis='y', colors='blue')
ax_right.tick_params(axis='y', colors='green')

lines_left, labels_left = ax_left.get_legend_handles_labels()
lines_right, labels_right = ax_right.get_legend_handles_labels()
ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc='upper right', ncol=2)


# Figure 2 - Localization and non-fixed noise

df = pd.read_csv(csv_path_non_fixed_loc)
print(f"Loaded {len(df)} rows from {csv_path_non_fixed_loc}")

interp_period = df.iloc[:, 0].to_numpy()

acc_full = df.iloc[:, 4].to_numpy()
nees_full = df.iloc[:, 5].to_numpy()

solve_time_inter = df.iloc[:, 6].to_numpy()*1e-6 # convert to s
solve_time_inter_iter = df.iloc[:, 8].to_numpy()*1e-6 # convert to s
acc_inter_all = df.iloc[:, 9].to_numpy()
acc_inter_est = df.iloc[:, 10].to_numpy()
nees_inter_all = df.iloc[:, 11].to_numpy()
nees_inter_est = df.iloc[:, 12].to_numpy()


plt.figure(figsize=(7, 6))
plt.xlabel("Interval of Non-Interpolated States")

#Name of plot
plt.title("Localization w/ Full Noise Model")

#Plot full vs interpolated solve time
plt.plot(interp_period, solve_time_full, label="Full Graph Solve Time", linestyle='--', marker=None, color='blue')
plt.plot(interp_period, solve_time_inter, label="Inter. Graph Solve Time", marker='o', color='blue')

#Plot full vs interpolated solve time per iteration
plt.plot(interp_period, solve_time_full_iter, label="Full Graph Iter. Time", linestyle='--', marker=None, color='lightblue')
plt.plot(interp_period, solve_time_inter_iter, label="Inter. Graph Iter. Time", marker='o', color='lightblue')

# put accuracy on same plot with different y-axis
plt.twinx()
plt.plot(interp_period, acc_full, label="Full Graph Acc.", linestyle='--', marker=None, color='green')
plt.plot(interp_period, acc_inter_all, label="Inter. Graph Acc. (All Times)", marker='o', color='green')
plt.plot(interp_period, acc_inter_est, label="Inter. Graph Acc. (Est. Times)", marker='o', color='lightgreen')

ax_right = plt.gca()
ax_right.set_ylim(0, 0.2)
ax_right.set_ylabel("Accuracy [m]")

ax_left = plt.gcf().axes[0]
ax_left.set_ylim(0, 0.25)
ax_left.set_ylabel("Time [s]")


#color code axis labels, ticks and lines to match plots
ax_left.yaxis.label.set_color('blue')
ax_right.yaxis.label.set_color('green')
ax_left.tick_params(axis='y', colors='blue')
ax_right.tick_params(axis='y', colors='green')

lines_left, labels_left = ax_left.get_legend_handles_labels()
lines_right, labels_right = ax_right.get_legend_handles_labels()
ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc='upper right', ncol=2)

# Figure 3 - SLAM and fixed noise


df = pd.read_csv(csv_path_fixed_slam)
print(f"Loaded {len(df)} rows from {csv_path_fixed_slam}")

interp_period = df.iloc[:, 0].to_numpy()

solve_time_full = df.iloc[:, 1].to_numpy()*1e-6 # convert to s
solve_time_full_iter = df.iloc[:, 3].to_numpy()*1e-6 # convert to s

#Take average of full solve time for clearer visualization
avg_solve_time_full = solve_time_full.mean()
solve_time_full = np.full_like(solve_time_full, avg_solve_time_full)

avg_solve_time_full_iter = solve_time_full_iter.mean()
solve_time_full_iter = np.full_like(solve_time_full_iter, avg_solve_time_full_iter)


acc_full = df.iloc[:, 4].to_numpy()
nees_full = df.iloc[:, 5].to_numpy()

solve_time_inter = df.iloc[:, 6].to_numpy()*1e-6 # convert to s
solve_time_inter_iter = df.iloc[:, 8].to_numpy()*1e-6 # convert to s
acc_inter_all = df.iloc[:, 9].to_numpy()
acc_inter_est = df.iloc[:, 10].to_numpy()
nees_inter_all = df.iloc[:, 11].to_numpy()
nees_inter_est = df.iloc[:, 12].to_numpy()


plt.figure(figsize=(7, 6))
plt.xlabel("Interval of Non-Interpolated States")

#Name of plot
plt.title("SLAM w/ Simplified Noise Model")

#Plot full vs interpolated solve time
plt.plot(interp_period, solve_time_full, label="Full Graph Solve Time", linestyle='--', marker=None, color='blue')
plt.plot(interp_period, solve_time_inter, label="Inter. Graph Solve Time", marker='o', color='blue')

#Plot full vs interpolated solve time per iteration
plt.plot(interp_period, solve_time_full_iter, label="Full Graph Iter. Time", linestyle='--', marker=None, color='lightblue')
plt.plot(interp_period, solve_time_inter_iter, label="Inter. Graph Iter. Time", marker='o', color='lightblue')

# put accuracy on same plot with different y-axis
plt.twinx()
plt.plot(interp_period, acc_full, label="Full Graph Acc.", linestyle='--', marker=None, color='green')
plt.plot(interp_period, acc_inter_all, label="Inter. Graph Acc. (All Times)", marker='o', color='green')
plt.plot(interp_period, acc_inter_est, label="Inter. Graph Acc. (Est. Times)", marker='o', color='lightgreen')

ax_right = plt.gca()
ax_right.set_ylim(0, 0.65)
ax_right.set_ylabel("Accuracy [m]")

ax_left = plt.gcf().axes[0]
ax_left.set_ylim(0, 0.7)
ax_left.set_ylabel("Time [s]")


#color code axis labels, ticks and lines to match plots
ax_left.yaxis.label.set_color('blue')
ax_right.yaxis.label.set_color('green')
ax_left.tick_params(axis='y', colors='blue')
ax_right.tick_params(axis='y', colors='green')

lines_left, labels_left = ax_left.get_legend_handles_labels()
lines_right, labels_right = ax_right.get_legend_handles_labels()
ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc='upper right', ncol=2)


# Figure 4 - SLAM and non-fixed noise


df = pd.read_csv(csv_path_non_fixed_slam)
print(f"Loaded {len(df)} rows from {csv_path_non_fixed_slam}")
interp_period = df.iloc[:, 0].to_numpy()

acc_full = df.iloc[:, 4].to_numpy()
nees_full = df.iloc[:, 5].to_numpy()

solve_time_inter = df.iloc[:, 6].to_numpy()*1e-6 # convert to s
solve_time_inter_iter = df.iloc[:, 8].to_numpy()*1e-6 # convert to s
acc_inter_all = df.iloc[:, 9].to_numpy()
acc_inter_est = df.iloc[:, 10].to_numpy()
nees_inter_all = df.iloc[:, 11].to_numpy()
nees_inter_est = df.iloc[:, 12].to_numpy()


plt.figure(figsize=(7, 6))
plt.xlabel("Interval of Non-Interpolated States")

#Name of plot
plt.title("SLAM w/ Full Noise Model")

#Plot full vs interpolated solve time
plt.plot(interp_period, solve_time_full, label="Full Graph Solve Time", linestyle='--', marker=None, color='blue')
plt.plot(interp_period, solve_time_inter, label="Inter. Graph Solve Time", marker='o', color='blue')

#Plot full vs interpolated solve time per iteration
plt.plot(interp_period, solve_time_full_iter, label="Full Graph Iter. Time", linestyle='--', marker=None, color='lightblue')
plt.plot(interp_period, solve_time_inter_iter, label="Inter. Graph Iter. Time", marker='o', color='lightblue')

# put accuracy on same plot with different y-axis
plt.twinx()
plt.plot(interp_period, acc_full, label="Full Graph Acc.", linestyle='--', marker=None, color='green')
plt.plot(interp_period, acc_inter_all, label="Inter. Graph Acc. (All Times)", marker='o', color='green')
plt.plot(interp_period, acc_inter_est, label="Inter. Graph Acc. (Est. Times)", marker='o', color='lightgreen')

ax_right = plt.gca()
ax_right.set_ylim(0, 0.65)
ax_right.set_ylabel("Accuracy [m]")

ax_left = plt.gcf().axes[0]
ax_left.set_ylim(0, 0.7)
ax_left.set_ylabel("Time [s]")


#color code axis labels, ticks and lines to match plots
ax_left.yaxis.label.set_color('blue')
ax_right.yaxis.label.set_color('green')
ax_left.tick_params(axis='y', colors='blue')
ax_right.tick_params(axis='y', colors='green')

lines_left, labels_left = ax_left.get_legend_handles_labels()
lines_right, labels_right = ax_right.get_legend_handles_labels()
ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc='upper right', ncol=2)

# Figure 5 - Localization NEES: Fixed vs Non-Fixed Noise
df_fixed_loc = pd.read_csv(csv_path_fixed_loc)
df_non_fixed_loc = pd.read_csv(csv_path_non_fixed_loc)

interp_period_fixed = df_fixed_loc.iloc[:, 0].to_numpy()
nees_full_fixed = df_fixed_loc.iloc[:, 5].to_numpy()
nees_inter_all_fixed = df_fixed_loc.iloc[:, 11].to_numpy()
nees_inter_est_fixed = df_fixed_loc.iloc[:, 12].to_numpy()

interp_period_non_fixed = df_non_fixed_loc.iloc[:, 0].to_numpy()
nees_full_non_fixed = df_non_fixed_loc.iloc[:, 5].to_numpy()
nees_inter_all_non_fixed = df_non_fixed_loc.iloc[:, 11].to_numpy()
nees_inter_est_non_fixed = df_non_fixed_loc.iloc[:, 12].to_numpy()

plt.figure(figsize=(7, 6))
plt.title("Localization NEES: Simplified vs Full Noise Model")
plt.xlabel("Interval of Non-Interpolated States")
plt.ylabel("NEES")

# Simplified noise model
plt.plot(interp_period_fixed, nees_full_fixed, label="Full Graph NEES", linestyle='--', color='tab:blue')
plt.plot(interp_period_fixed, nees_inter_all_fixed, label="Simpl. Noise Model (All Times)", marker='o', color='tab:blue')
plt.plot(interp_period_fixed, nees_inter_est_fixed, label="Simpl. Noise Model (Est. Times)", marker='s', color='lightblue')

# Non-fixed noise model
plt.plot(interp_period_non_fixed, nees_inter_all_non_fixed, label="Full Noise Model (All Times)", marker='o', color='tab:orange')
plt.plot(interp_period_non_fixed, nees_inter_est_non_fixed, label="Full Noise Model (Est. Times)", marker='s', color='#ffcc99')

plt.legend(loc='upper left')


# Figure 5 - SLAM NEES: Fixed vs Non-Fixed Noise
df_fixed_loc = pd.read_csv(csv_path_fixed_slam)
df_non_fixed_loc = pd.read_csv(csv_path_non_fixed_slam)

interp_period_fixed = df_fixed_loc.iloc[:, 0].to_numpy()
nees_full_fixed = df_fixed_loc.iloc[:, 5].to_numpy()
nees_inter_all_fixed = df_fixed_loc.iloc[:, 11].to_numpy()
nees_inter_est_fixed = df_fixed_loc.iloc[:, 12].to_numpy()

interp_period_non_fixed = df_non_fixed_loc.iloc[:, 0].to_numpy()
nees_full_non_fixed = df_non_fixed_loc.iloc[:, 5].to_numpy()
nees_inter_all_non_fixed = df_non_fixed_loc.iloc[:, 11].to_numpy()
nees_inter_est_non_fixed = df_non_fixed_loc.iloc[:, 12].to_numpy()

plt.figure(figsize=(7, 6))
plt.title("SLAM NEES: Simplified vs Full Noise Model")
plt.xlabel("Interval of Non-Interpolated States")
plt.ylabel("NEES")

# Simplified noise model
plt.plot(interp_period_fixed, nees_full_fixed, label="Full Graph NEES", linestyle='--', color='tab:blue')
plt.plot(interp_period_fixed, nees_inter_all_fixed, label="Simpl. Noise Model (All Times)", marker='o', color='tab:blue')
plt.plot(interp_period_fixed, nees_inter_est_fixed, label="Simpl. Noise Model (Est. Times)", marker='s', color='lightblue')

# Non-fixed noise model
plt.plot(interp_period_non_fixed, nees_inter_all_non_fixed, label="Full Noise Model (All Times)", marker='o', color='tab:orange')
plt.plot(interp_period_non_fixed, nees_inter_est_non_fixed, label="Full Noise Model (Est. Times)", marker='s', color='#ffcc99')

plt.legend(loc='upper left')

# Save all six plots as SVGs
output_dir = Path("../results/woods_timing_results/plots")
output_dir.mkdir(parents=True, exist_ok=True)

fig_filenames = {
    1: "loc_simplified_timing.svg",
    2: "loc_full_timing.svg",
    3: "slam_simplified_timing.svg",
    4: "slam_full_timing.svg",
    5: "loc_nees_compare.svg",
    6: "slam_nees_compare.svg",
}

for num, fname in fig_filenames.items():
    fig = plt.figure(num)
    fig.savefig(output_dir / fname, format="svg", bbox_inches="tight")


plt.show()