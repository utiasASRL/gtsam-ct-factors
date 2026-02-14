%% Get groundtruth from raw data
load /home/daniel/gtsam/gtsam_repo/examples/Data/starryNight.mat;
%{
starryNight.mat:
theta_vk_i : a 3 × K matrix where the k th column is the axis-angle representation of the groundtruth value of
C_{vk,i}
r_i_vk_i : a 3 × K matrix where the kth column is the groundtruth value of r_i^{vk,i} [m]
t : a 1 × K matrix of time values t(k) [s]
w_vk_vk_i : a 3 × K matrix where the k th column is the measured rotational velocity, ω_vk^{vk,i} [rad/s]
w_var : a 3 × 1 matrix of the computed variances (based on groundtruth) of the rotational speeds [rad^2 /s^2 ]
v_vk_vk i : a 3 × K matrix where the k th column is the measured translational velocity, v_vk^{vk,i} [m/s]
v_var : a 3 × 1 matrix of the computed variances (based on groundtruth) of the translational speeds [m^2 /s^2 ]
rho_i_pj_i : a 3 × 20 matrix where the jth column is the position of feature j, ρ_i^{pj, i} [m]
y_k_j : a 4 × K × 20 array of observations, y_k^j [pixels]. All components of y k j(:,k,j) will be −1 if the
observation is invalid.
y_var : a 4 × 1 matrix of the computed variances (based on groundtruth) of the stereo measurements [pixels^2 ]
C_c_v : a 3 × 3 matrix giving the rotation from the vehicle frame to the camera frame, C_{cv}
rho_v_c_v : a 3 × 1 matrix giving the translation from the vehicle frame to the camera frame, ρ_v^{cv} [m]
fu : the stereo camera’s horizontal focal length, fu [pixels]
fv : the stereo camera’s vertical focal length, fv [pixels]
cu : the stereo camera’s horizontal optical center, cu [pixels]
cv : the stereo camera’s vertical optical center, cv [pixels]
b : the stereo camera baseline, b [m]
%}

% get groundtruth for this section
k1 = 1215;
k2 = 1714;
n = k2 - k1+1;
r_gt = r_i_vk_i(:,k1:k2);
theta_gt = theta_vk_i(:,k1:k2);
C_gt = zeros(3,3,n);
for i = 1:n
    C_gt(:,:,i) = psitoC(theta_gt(:,i))';
end

%% load GTSAM results
fp = '/home/daniel/gtsam/gtsam_repo/results/';  % filename prefix
pose_interval = 5;
[C_array_odom, t_array_odom] = load_poses([fp 'starry_night_results/starryNightOdom_poses.csv']);
[C_array_wnoa, t_array_wnoa] = load_poses([fp 'starry_night_results/starryNightWNOA_poses.csv']);
[C_array_interpolated, t_array_interpolated] = load_poses([fp 'starry_night_results/starryNightInterp_poses.csv']);
l_array = load_landmarks([fp 'starry_night_results_landmarks.csv']);

% get marginals (covariances)
[sigmas_3_all] = load_marginals([fp 'starry_night_results/starryNightOdom_marginals.csv']);
[sigmas_3_wnoa] = load_marginals([fp 'starry_night_results/starryNightWNOA_marginals.csv']);
[sigmas_3_interpolated] = load_marginals([fp 'starry_night_results/starryNightInterp_marginals.csv']);

%% plotting, same format as a3.m
fig = figure('Position', [660   187   756   667]);
set(gcf, 'Color', 'w');
scatter3(l_array(1,:), l_array(2,:), l_array(3,:), 'LineWidth', 1.5, 'MarkerEdgeColor', [0 0.3 0], 'DisplayName', 'Landmarks')
hold on
plot3(r_gt(1,:), r_gt(2,:), r_gt(3,:), '.-', 'Color', [0.2, 0.6, 0.1], 'MarkerSize', 10, 'LineWidth', 0.5, 'DisplayName', 'Ground-Truth Trajectory')  % ground truth
plot3(t_array_wnoa(1,:), t_array_wnoa(2,:), t_array_wnoa(3,:), 'm.-','MarkerSize', 10, 'LineWidth', 0.5, 'DisplayName', 'Optimized Trajectory (WNOA and Measurements, without Interpolation)')
plot3(t_array_interpolated(1,:), t_array_interpolated(2,:), t_array_interpolated(3,:), '.-', 'LineWidth', 0.5, 'MarkerSize', 10, 'Color', 'b', 'DisplayName', 'Optimized Trajectory (WNOA and Measurements, with Interpolation)')
scatter3(t_array_interpolated(1,1:end), t_array_interpolated(2,1:end), t_array_interpolated(3,1:end), 100, '.', 'LineWidth', 2, 'MarkerEdgeColor', [0.3 0.8 1], 'DisplayName', 'Interpolated States')
scatter3(t_array_interpolated(1,1:pose_interval:end), t_array_interpolated(2,1:pose_interval:end), t_array_interpolated(3,1:pose_interval:end), 200, '.', 'LineWidth', 2, 'MarkerEdgeColor', 'b', 'HandleVisibility', 'off')

xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]')
axis equal
legend('Location', 'northwest')
% title('Visualizing the Effects of Interpolation on the Starry Night Dataset', 'FontWeight','normal', 'FontSize',14)
campos([ -3.5882   -6.7749    5.5577])
% export_fig 'starryNight.png' -m5

%% errors
% interp_end_idx = size(t_array_interpolated,2);  % plot for the whole solution
interp_end_idx = 200;  % plot just a subsection for clarity in paper

plot_type_choices = {'wnoa', 'interpolated'};
plot_type = 'interpolated';
assert(ismember(plot_type, plot_type_choices));
if strcmp(plot_type, 'wnoa')
    % wnoa errors
    error_r = t_array_wnoa(:,1:interp_end_idx) - r_gt(:,1:interp_end_idx);
    error_theta = compute_error_theta(C_array_wnoa, C_gt(:,:,1:interp_end_idx));
    end_idx_diff = size(t_array_wnoa,2) - interp_end_idx;
    errors_to_plot = [error_r; error_theta];
    sigmas_to_plot = sigmas_3_wnoa(:,1:interp_end_idx);
    marker_color = [0.9 0 0.9];
    plot_title = 'Pose Errors, without Interpolation';
else
    % interpolated errors
    error_r = t_array_interpolated(:,1:interp_end_idx) - r_gt(:,1:interp_end_idx);
    error_theta = compute_error_theta(C_array_interpolated, C_gt(:,:,1:interp_end_idx));
    end_idx_diff = size(t_array_interpolated,2) - interp_end_idx;
    errors_to_plot = [error_r; error_theta];
    sigmas_to_plot = sigmas_3_interpolated(:,1:interp_end_idx);
    marker_color = [0.2 0.7 0.8];
    plot_title = 'Pose Errors, with Interpolation';
end

lim_trans = 0.55;
lim_rot = 0.25;
fig = figure('Position', [950   305   507   752]);
set(gcf, 'Color', 'w');
tiledlayout(6, 1, 'Padding', 'none', 'TileSpacing', 'compact');
y_label_array = {'Error in x [m]', 'Error in y [m]', 'Error in z [m]', 'Error in \theta_x [rad]', 'Error in \theta_y [rad]', 'Error in \theta_z [rad]'};
for i = 1:6
    nexttile(i);
    scatter(t(k1:1:k2-end_idx_diff), errors_to_plot(i,1:1:end), 100, '.', 'MarkerEdgeColor', marker_color);
    hold on
    x2 = [t(k1:k2-end_idx_diff), fliplr(t(k1:k2-end_idx_diff))];
    inBetween = [sigmas_to_plot(i,:), fliplr(-sigmas_to_plot(i,:))];
    fill(x2, inBetween, 'b', 'LineStyle', '--', 'EdgeColor', 'none', 'FaceAlpha', 0.2, 'HandleVisibility', 'off');  % shade covariance envelope
    if strcmp(plot_type, 'interpolated')
        scatter(t(k1:pose_interval:k2-end_idx_diff), errors_to_plot(i,1:pose_interval:end), 100, '.', 'MarkerEdgeColor', 'blue');  % with interpolation
    end
    xlabel('Time [s]');
    ylabel(y_label_array{i})
    if i < 4
        ylim([-lim_trans lim_trans])
    else
        ylim([-lim_rot lim_rot])
    end
    xlim([111.5 inf])
    grid on
end
nexttile(1)
title(plot_title, 'FontWeight','normal', 'FontSize',14)
% export_fig 'starryNightErrPlot2.png' -m5
%% just comparing covariance between interpolation and no interpolation
% This plot did not make it to the paper.
interp_end_idx = size(t_array_interpolated,2);
end_idx_diff = size(t_array_interpolated,2) - interp_end_idx;
figure
set(gcf, 'Color', 'w');
tiledlayout(6, 1, 'Padding', 'none', 'TileSpacing', 'compact');
y_label_array = {'\sigma_x', '\sigma_y', '\sigma_z', '\sigma_\theta_x', '\sigma_\theta_y', '\sigma_\theta_z'};
lim_trans = 3.0;
lim_rot = 1.2;
for i = 1:6
    nexttile(i);
    plot(t(k1:k2), sigmas_3_wnoa(i,:), '-', 'LineWidth', 1.5, 'Color', 'm', 'DisplayName', 'No Interpolation');
    hold on
    plot(t(k1:k2-end_idx_diff), sigmas_3_interpolated(i,:), '.-', 'LineWidth', 1.5, 'Color', 'blue', 'DisplayName', 'With Interpolation');
    scatter(t(k1:pose_interval:k2), sigmas_3_interpolated(i,1:pose_interval:end), 'o', 'LineWidth', 2, 'MarkerEdgeColor', 'blue', 'DisplayName', 'States in main solve');
    xlabel('time (s)');
    ylabel(y_label_array{i})
    if i < 4
        ylim([0 lim_trans])
    else
        ylim([0 lim_rot])
    end
end
nexttile(1)
title('Covariance plot for WNOA and measurements, with and without interpolation')
nexttile(6)
legend('Location','southoutside')
% result: the two covariances are almost identical, as expected.

%%
function C = psitoC(psi)

validateattributes(psi,{'double'},{'nrows',3})

psinorm = norm(psi);
C = cos(psinorm) * eye(3) + (1-cos(psinorm)) * (psi/psinorm) * (psi/psinorm)' - sin(psinorm)*hat(psi/psinorm);

end

function [C_array, t_array] = load_poses(filepath)
result_poses = readtable(filepath);
n = size(result_poses,1);
C_array = zeros(3,3,n);
t_array = zeros(3,n);
for i = 1:n
    C_array(:,:,i) = eul2rotm([result_poses.yaw(i) result_poses.pitch(i) result_poses.roll(i)], "ZYX");
    t_array(:,i) = [result_poses.x(i); result_poses.y(i); result_poses.z(i)];
end
end

function [t_array] = load_landmarks(filepath)
result_landmarks = readtable(filepath);
n = size(result_landmarks,1);
t_array = zeros(3,n);
for i = 1:size(result_landmarks,1)
    t_array(:,i) = [result_landmarks.x(i); result_landmarks.y(i); result_landmarks.z(i)];
end
end

function [sigmas_3] = load_marginals(filepath)
marg_input = readmatrix(filepath);
% each row is a flattened cov matrix
n = size(marg_input, 1);
P = reshape(marg_input', 6, 6, n);
sigmas_3 = zeros(6, n);
for i = 1:n
    try chol(P(:,:,i));
    catch ME
        disp('Matrix is not symmetric positive definite')
    end
    sigmas_3(:,i) = 3*sqrt(diag(P(:,:,i)));
end
end

function error_theta = compute_error_theta(C_est, C_gt)
n = size(C_gt, 3);
error_theta = zeros(3, n);
for k = 1:n
    dC = eye(3) - C_est(:,:,k) * C_gt(:,:,k)';
    error_theta(:,k) = [-dC(2,3); dC(1,3); -dC(1,2)];
end
end
