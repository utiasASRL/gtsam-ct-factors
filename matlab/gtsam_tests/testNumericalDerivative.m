import gtsam.*;

%% test_numericalDerivative11_scalar
% Test function of one variable
h = @(x) x^2;
x = 3.0;
% Analytical derivative: dh/dx = 2x
analytical_derivative = 2.0 * x;
% Compute numerical derivative
numerical_derivative = numericalDerivative.numericalDerivative11(h, x);
EQUALITY('test_numericalDerivative11_scalar', analytical_derivative, numerical_derivative, 1e-5);

%% test_numericalDerivative11_vector
% Test function of one vector variable
h = @(x) x.^2;
x = [1.0; 2.0; 3.0];
% Analytical derivative: dh/dx = 2x
analytical_derivative = diag(2.0 * x);
numerical_derivative = numericalDerivative.numericalDerivative11(h, x);
EQUALITY('test_numericalDerivative11_vector', analytical_derivative, numerical_derivative, 1e-5);

%% test_numericalDerivative21
% Test function of two variables, derivative with respect to first variable
h = @(x1, x2) x1 * sin(x2);
x1 = 2.0;
x2 = pi / 4;
% Analytical derivative: dh/dx1 = sin(x2)
analytical_derivative = sin(x2);
numerical_derivative = numericalDerivative.numericalDerivative21(h, x1, x2);
EQUALITY('test_numericalDerivative21', analytical_derivative, numerical_derivative, 1e-5);

%% test_numericalDerivative22
% Test function of two variables, derivative with respect to second variable
h = @(x1, x2) x1 * sin(x2);
x1 = 2.0;
x2 = pi / 4;
% Analytical derivative: dh/dx2 = x1 * cos(x2)
analytical_derivative = x1 * cos(x2);
numerical_derivative = numericalDerivative.numericalDerivative22(h, x1, x2);
EQUALITY('test_numericalDerivative22', analytical_derivative, numerical_derivative, 1e-5);

%% test_numericalDerivative33
% Test function of three variables, derivative with respect to third variable
h = @(x1, x2, x3) x1 * x2 + exp(x3);
x1 = 1.0;
x2 = 2.0;
x3 = 0.5;
% Analytical derivative: dh/dx3 = exp(x3)
analytical_derivative = exp(x3);
numerical_derivative = numericalDerivative.numericalDerivative33(h, x1, x2, x3);
EQUALITY('test_numericalDerivative33', analytical_derivative, numerical_derivative, 1e-5);

%% test_numericalDerivative_with_pose
% Test function with manifold and vector inputs
h = @(pose, point) pose.transformFrom(point);

% Values from testPose3.cpp
P = Point3(0.2, 0.7, -2);
R = Rot3.Rodrigues(0.3, 0, 0);
P2 = Point3(3.5, -8.2, 4.2);
T = Pose3(R, P2);

% Compute analytical Jacobians manually, as in C++
R = T.rotation().matrix();
skew = [  0      -P(3)   P(2);
	       P(3)    0      -P(1);
        -P(2)  P(1)     0   ];
analytic_H1 = zeros(3,6);
analytic_H1(:,1:3) = R * (-skew); % leftCols<3>()
analytic_H1(:,4:6) = R;           % rightCols<3>()
analytic_H2 = R;

numerical_H1 = numericalDerivative.numericalDerivative21(h, T, P);
numerical_H2 = numericalDerivative.numericalDerivative22(h, T, P);

EQUALITY('test_numericalDerivative_with_pose_H1', analytic_H1, numerical_H1, 1e-5);
EQUALITY('test_numericalDerivative_with_pose_H2', analytic_H2, numerical_H2, 1e-5);