%% compare with gt
load /home/daniel/Dropbox/Coursework/year_1/AER1513/assignment/3/dataset3.mat
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
pose_interval = 10;
[C_array_odom, t_array_odom] = load_poses([fp 'starry_night_results_poses_no_wnoa_with_odom_with_meas_odom_init_pose_interval_1_max_pose_1000.csv']);
[C_array_wnoa, t_array_wnoa] = load_poses([fp 'starry_night_results_poses_with_wnoa_no_odom_with_meas_odom_init_pose_interval_1_max_pose_1000.csv']);
[C_array_interval, t_array_interval] = load_poses([fp 'starry_night_results_poses_with_wnoa_no_odom_with_meas_gt_init_pose_interval_' num2str(pose_interval) '_max_pose_1000.csv']);
[C_array_interpolated, t_array_interpolated] = load_poses([fp 'starry_night_results_poses_with_wnoa_no_odom_with_meas_gt_init_pose_interval_' num2str(pose_interval) '_max_pose_1000_interpolated.csv']);
[C_array_dr, t_array_dr] = load_poses([fp 'starry_night_results_poses_dr_no_wnoa_with_odom_with_meas_odom_init_pose_interval_1_max_pose_1000.csv']);
l_array = load_landmarks('/home/daniel/gtsam/gtsam_repo/results/starry_night_results_landmarks.csv');

% get marginals
marg_input = readmatrix([fp 'starry_night_results_marginals_with_wnoa_no_odom_with_meas_odom_init_pose_interval_1_max_pose_1000.csv']);
% each row is a flattened cov matrix
P = reshape(marg_input', 6, 6, []);
sigmas_3 = zeros(6, n);
for i = 1:n
    sigmas_3(:,i) = 3*sqrt(diag(P(:,:,i)));
end

%% load MATLAB results
load /home/daniel/Dropbox/Coursework/year_1/AER1513/assignment/3/dataset3_results.mat r_k C_k  % dead reckoning results
C_k_trans = pagetranspose(C_k);
%% plotting, same format as a3.m
figure
scatter3(l_array(1,:), l_array(2,:), l_array(3,:), 'DisplayName', 'Landmarks')
hold on
plot3(t_array_wnoa(1,:), t_array_wnoa(2,:), t_array_wnoa(3,:), 'LineWidth', 2, 'DisplayName', 'WNOA (no odometry) solution')
plot3(t_array_odom(1,:), t_array_odom(2,:), t_array_odom(3,:), 'LineWidth', 2, 'DisplayName', 'Odometry solution')
% plot3(t_array_interval(1,:), t_array_interval(2,:), t_array_interval(3,:), '.-', 'MarkerSize', 10, 'Color', 'r', 'DisplayName', 'All states in main solve')
% plot3(t_array_interpolated(1,:), t_array_interpolated(2,:), t_array_interpolated(3,:), '.-', 'MarkerSize', 10, 'Color', 'b', 'DisplayName', 'Sparse states in main solve, interpolated afterwards')
% scatter3(t_array_interval(1,1:pose_interval:end), t_array_interval(2,1:pose_interval:end), t_array_interval(3,1:pose_interval:end), 60, 'o', 'LineWidth', 2, 'MarkerEdgeColor', 'r', 'HandleVisibility', 'off')
% scatter3(t_array_interpolated(1,1:pose_interval:end), t_array_interpolated(2,1:pose_interval:end), t_array_interpolated(3,1:pose_interval:end), 60, 'o', 'LineWidth', 2, 'MarkerEdgeColor', 'b', 'HandleVisibility', 'off')

plot3(r_gt(1,:), r_gt(2,:), r_gt(3,:), 'LineWidth', 2, 'DisplayName', 'Groundtruth')  % ground truth
% scatter3(r_gt(1,1:pose_interval:end), r_gt(2,1:pose_interval:end), r_gt(3,1:pose_interval:end), 60, 'o', 'LineWidth', 1, 'MarkerEdgeColor', 'black', 'HandleVisibility', 'off')
% plot3(t_array_dr(1,:), t_array_dr(2,:), t_array_dr(3,:), 'DisplayName', 'Dead reckoning')  % dead reckoning
% plot3(r_k(1,:), r_k(2,:), r_k(3,:), 'DisplayName', 'Dead reckoning MATLAB')  % dead reckoning
xlabel('x'); ylabel('y'); zlabel('z')
axis equal
legend('Location', 'south')
title('A3 with GTSAM')
%% errors
error_r = t_array_wnoa - r_gt;
error_theta = compute_error_theta(C_array_wnoa, C_gt);
end_idx_diff = 0;
errors_to_plot = [error_r; error_theta];

% interp_end_idx = idivide(size(t_array_interval,2)-1, int16(pose_interval)) * pose_interval + 1;
% end_idx_diff = size(t_array_interval,2) - interp_end_idx;
% error_r_interp = t_array_interval(:,1:interp_end_idx) - t_array_interpolated;
% error_theta_interp = compute_error_theta(C_array_interval(:,:,1:interp_end_idx), C_array_interpolated);
% errors_to_plot = [error_r_interp; error_theta_interp];

% error_r_dr = t_array - r_k;
% error_theta_dr = compute_error_theta(C_array, C_k_trans);
% errors_to_plot = [error_r_dr; error_theta_dr];

lim_trans = 0.15;
lim_rot = 0.15;
% lim_trans = 0.3;
% lim_rot = 0.3;
figure
set(gcf, 'Color', 'w');
tiledlayout(6, 1, 'Padding', 'none', 'TileSpacing', 'compact');
y_label_array = {'error in x', 'error in y', 'error in z', 'error in \theta_x', 'error in \theta_y', 'error in theta_z'};
for i = 1:6
    nexttile(i);
    % plot(t(k1:k2-end_idx_diff), errors_to_plot(i,:), '.', 'LineWidth', 1, 'Color', 'b');
    plot(t(k1:k2-end_idx_diff), errors_to_plot(i,:), '-', 'LineWidth', 1, 'Color', 'b');
    hold on
    % scatter(t(k1:pose_interval:k2-end_idx_diff), errors_to_plot(i,1:pose_interval:end), 'o', 'MarkerEdgeColor', 'blue');
    plot(t(k1:k2), sigmas_3(i,:), '--');
    plot(t(k1:k2), -sigmas_3(i,:), '--');
    xlabel('time (s)');
    ylabel(y_label_array{i})
    % legend
    if i < 4
        ylim([-lim_trans lim_trans])
    else
        ylim([-lim_rot lim_rot])
    end
end

%% old code - plot errors
lim_trans = 0.15;
lim_rot = 0.3;
figure
subplot(3,1,1)
title('x error')
xlabel('time (s)')
ylabel('error in x (m)')
hold on
plot(t(k1:k2), error_r(1,:));
plot(t(k1:k2), sigmas_3(1,:), '--');
plot(t(k1:k2), -sigmas_3(1,:), '--');
ylim([-lim_trans lim_trans])

subplot(3,1,2)
title('y error')
xlabel('time (s)')
ylabel('error in y (m)')
hold on
plot(t(k1:k2), error_r(2,:));
plot(t(k1:k2), sigmas_3(2,:), '--');
plot(t(k1:k2), -sigmas_3(2,:), '--');
ylim([-lim_trans lim_trans])

subplot(3,1,3)
title('z error')
xlabel('time (s)')
ylabel('error in z (m)')
hold on
plot(t(k1:k2), error_r(3,:));
plot(t(k1:k2), sigmas_3(3,:), '--');
plot(t(k1:k2), -sigmas_3(3,:), '--');
ylim([-lim_trans lim_trans])

figure
subplot(3,1,1)
title('\theta_x error')
xlabel('time (s)')
ylabel('error in \theta_x (rad)')
hold on
plot(t(k1:k2), error_theta(1,:));
plot(t(k1:k2), sigmas_3(4,:), '--');
plot(t(k1:k2), -sigmas_3(4,:), '--');
ylim([-lim_rot lim_rot])

subplot(3,1,2)
title('\theta_y error')
xlabel('time (s)')
ylabel('error in \theta_y (rad)')
hold on
plot(t(k1:k2), error_theta(2,:));
plot(t(k1:k2), sigmas_3(5,:), '--');
plot(t(k1:k2), -sigmas_3(5,:), '--');
ylim([-lim_rot lim_rot])

subplot(3,1,3)
title('\theta_z error')
xlabel('time (s)')
ylabel('error in \theta_z (rad)')
hold on
plot(t(k1:k2), error_theta(3,:));
plot(t(k1:k2), sigmas_3(6,:), '--');
plot(t(k1:k2), -sigmas_3(6,:), '--');
ylim([-lim_rot lim_rot])

%% old code using gtsam matlab toolbox
% for instructions, see https://github.com/borglab/gtsam/blob/develop/matlab/README.md
% don't know how to set up paths properly yet... seems like it only needs gtsamDebug
% addpath('~/gtsam/')
addpath('~/gtsamDebug/')
format compact
% addpath('~/gtsamDebug/gtsam_examples/')

[graph_init, initial] = gtsam.readG2o('/home/daniel/gtsam/gtsam_repo/build/starry_night_initial.g2o', true);
[graph_opt, optimized] = gtsam.readG2o('/home/daniel/gtsam/gtsam_repo/build/starry_night_optimized.g2o', true);
%% get solution poses
C_array = zeros(3,3,n);
t_array = zeros(3,n);
for i = 1:n
    pose3 = optimized.atPose3(i);
    C_array(:,:,i) = pose3.rotation().matrix()';
    t_array(:,i) = pose3.translation();
end

% get dr poses
C_array_dr = zeros(3,3,n);
t_array_dr = zeros(3,n);
for i = 1:n
    pose3_dr = initial.atPose3(i);
    C_array_dr(:,:,i) = pose3_dr.rotation().matrix()';
    t_array_dr(:,i) = pose3_dr.translation();
end

% get landmarks - just for double checking. They shouldn't change compared
% to the gt
m = 20;  % num of landmarks
l_array = zeros(3,m);
for j = 1:m
    l_array(:,j) = optimized.atPoint3(gtsam.symbol('l', j-1));
end
%% plotting
figure
gtsam.plot3DPoints(initial, 'oblack');  % Red stars for landmarks
gtsam.plot3DTrajectory(initial, 'g');      % Green for initial guess
gtsam.plot3DTrajectory(optimized, 'b');      % Blue for optimized
hold on
grid on;
view(3);
axis equal;
xlabel('x'); ylabel('y'); zlabel('z')
legend
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

function error_theta = compute_error_theta(C_est, C_gt)
n = size(C_gt, 3);
error_theta = zeros(3, n);
for k = 1:n
    dC = eye(3) - C_est(:,:,k) * C_gt(:,:,k)';
    error_theta(:,k) = [-dC(2,3); dC(1,3); -dC(1,2)];
end
end
