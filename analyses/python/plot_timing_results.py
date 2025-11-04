import numpy as np
import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt

csv_path = Path("../results/woods_timing_results/loc_parallel_non_fixed_noise.csv")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows from {csv_path}")

interp_period = df.iloc[:, 0].to_numpy()

solve_time_full = df.iloc[:, 1].to_numpy()*1e-6 # convert to s

#Take average of full solve time for clearer visualization
avg_solve_time_full = solve_time_full.mean()
solve_time_full = np.full_like(solve_time_full, avg_solve_time_full)

solve_time_inter = df.iloc[:, 4].to_numpy()*1e-6 # convert to s
acc_full = df.iloc[:, 3].to_numpy()
acc_inter = df.iloc[:, 8].to_numpy()

plt.figure(figsize=(8, 6))
plt.xlabel("Frequency of Non-Interpolated States")

#Name of plot
plt.title("SLAM, Sequential, Fixed Noise")

plt.plot(interp_period, solve_time_full, label="Full Solve Time", linestyle='--', marker=None, color='blue')
plt.plot(interp_period, solve_time_inter, label="Interpolated Solve Time", marker='o', color='blue')

# put accuracy on same plot with different y-axis
plt.twinx()
plt.plot(interp_period, acc_full, label="Full Accuracy", linestyle='--', marker=None, color='green')
plt.plot(interp_period, acc_inter, label="Interpolated Accuracy", marker='o', color='green')

ax_right = plt.gca()
ax_right.set_ylim(0, 0.35)
ax_right.set_ylabel("Accuracy in m")

ax_left = plt.gcf().axes[0]
ax_left.set_ylim(0, 1.0)
ax_left.set_ylabel("Solve Time in s")


#color code axis labels, ticks and lines to match plots
ax_left.yaxis.label.set_color('blue')
ax_right.yaxis.label.set_color('green')
ax_left.tick_params(axis='y', colors='blue')
ax_right.tick_params(axis='y', colors='green')

lines_left, labels_left = ax_left.get_legend_handles_labels()
lines_right, labels_right = ax_right.get_legend_handles_labels()
ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc='lower right')


plt.show()