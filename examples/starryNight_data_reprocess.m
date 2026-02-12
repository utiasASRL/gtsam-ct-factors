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

%% trim data down to k1 to k2
k1 = 1215;
k2 = 1714;
K = k2-k1+1;
t_trim = t(:,k1:k2);
v_trim = v_vk_vk_i(:,k1:k2);
w_trim = w_vk_vk_i(:,k1:k2);
y_trim = y_k_j(:,k1:k2,:);
y = reshape(permute(y_trim, [1 3 2]), [], size(y_trim,2))';  % go from 4 x K x 20 to K x 80

%% get rotation matrix gt
% this is T_iv, i.e., in the inertial (world) frame
T_gt = zeros(4,4,K);
T_gt(4,4,:) = 1;
T_gt_reshaped = zeros(K, 16);
for k = 1:K
    T_gt(1:3,1:3,k) = psitoC(theta_vk_i(:,k1+k-1))';
    T_gt(1:3,4,k) = r_i_vk_i(:,k1+k-1);
    T_gt_reshaped(k,:) = reshape(T_gt(:,:,k)', 1, []);
end
%%
C_v_c = C_c_v';
% y_var is in the form for [ul, vl, ur, vr]
% but we want it in the form of [ul ur vl vr]
y_var_reformat = [y_var(1) y_var(3) y_var(2) y_var(4)];

%% write data
data_gt = T_gt_reshaped;
data_landmark_pos = rho_i_pj_i';
writematrix([t_trim' w_trim' v_trim'], 'starryNightInput.txt', 'Delimiter',' ')
writematrix(y, 'starryNightMeas.txt', 'Delimiter',' ')
writematrix(data_gt, 'starryNightGroundtruth.txt', 'Delimiter',' ')
writematrix(data_landmark_pos, 'starryNightLandmarkPos.txt', 'Delimiter',' ')
%% write metadata
metadata_name = 'starryNightMetadata.txt';
writematrix([fu fv 0 cu cv b], metadata_name, 'Delimiter',' ')  % skew is 0
writematrix(reshape(C_v_c', 1, []), metadata_name, 'WriteMode', 'append', 'Delimiter',' ')
writematrix(rho_v_c_v', metadata_name, 'WriteMode', 'append', 'Delimiter',' ')
writematrix([w_var' v_var'], metadata_name,  'WriteMode', 'append', 'Delimiter',' ')
writematrix(y_var_reformat, metadata_name, 'WriteMode', 'append', 'Delimiter',' ')
