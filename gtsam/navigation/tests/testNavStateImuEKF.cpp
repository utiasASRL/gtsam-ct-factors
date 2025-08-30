/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 *
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file testNavStateImuEKF.cpp
 * @brief Unit test for NavStateImuEKF, as well as dynamics used.
 * @date April 26, 2025
 * @authors Scott Baker, Matt Kielo, Frank Dellaert
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/NavStateImuEKF.h>
#include <gtsam/navigation/PreintegrationParams.h>

using namespace gtsam;

namespace nontrivial_navstate_example {
// A nontrivial NavState
Rot3 R0 = Rot3::RzRyRx(0.1, -0.2, 0.3);
Point3 p0(0.5, -0.4, 0.3);
Vector3 v0(0.2, -0.1, 0.05);
NavState X0(R0, p0, v0);

// Controls and parameters
Vector3 gyro(0.3, -0.2, 0.1);
Vector3 accel(0.5, -0.3, 0.2);
auto params = PreintegrationParams::MakeSharedU(9.81);
}  // namespace nontrivial_navstate_example

TEST(NavStateImuEKF, DefaultProcessNoiseFromParams) {
  using namespace nontrivial_navstate_example;

  // GIVEN params with specific covariances
  auto params = PreintegrationParams::MakeSharedU(9.81);
  Matrix3 Cg = (Matrix3() << 0.01, 0, 0, 0, 0.02, 0, 0, 0, 0.03).finished();
  Matrix3 Ci = (Matrix3() << 0.001, 0, 0, 0, 0.002, 0, 0, 0, 0.003).finished();
  Matrix3 Ca = (Matrix3() << 0.1, 0, 0, 0, 0.2, 0, 0, 0, 0.3).finished();
  params->setGyroscopeCovariance(Cg);
  params->setIntegrationCovariance(Ci);
  params->setAccelerometerCovariance(Ca);

  Matrix9 P0 = I_9x9 * 0.01;
  NavStateImuEKF ekf(X0, P0, params);

  Matrix9 Q = Matrix9::Zero();
  Q.block<3, 3>(0, 0) = Cg;
  Q.block<3, 3>(3, 3) = Ci;
  Q.block<3, 3>(6, 6) = Ca;
  EXPECT(assert_equal(Q, ekf.processNoise(), 1e-12));
}

TEST(NavStateImuEKF, DynamicsJacobian) {
  using namespace nontrivial_navstate_example;

  // Check the Jacobian of navStateImuDynamics
  double dt = 0.01;
  Matrix9 H;
  NavState U =
      navStateImuDynamics(X0, gyro, accel, dt, params->getGravity(), H);
  (void)U;
  std::function<NavState(const NavState&)> f =
      [&](const NavState& Xq) -> NavState {
    return navStateImuDynamics(Xq, gyro, accel, dt, params->getGravity());
  };
  Matrix9 Hnum = numericalDerivative11(f, X0);

  EXPECT(assert_equal(Hnum, H, 1e-6));
}

TEST(NavStateImuEKF, PredictMatchesExplicitIntegration) {
  using namespace nontrivial_navstate_example;
  double dt = 0.02;

  // Explicit integration as per Derek's code (from raw inputs)
  Rot3 dR = Rot3::Expmap(gyro * dt);
  Rot3 R_new = R0.compose(dR);  // R_new = R * dR
  Vector3 a_world = R0 * accel + params->n_gravity;
  Vector3 v_new = v0 + a_world * dt;
  Point3 p_new = p0 + v0 * dt + a_world * (0.5 * dt * dt);
  NavState X_explicit(R_new, p_new, v_new);

  // Increment-based integration should match exactly
  auto params = PreintegrationParams::MakeSharedU(9.81);
  NavStateImuEKF ekf(X0, I_9x9 * 1e-3, params);
  NavState U = navStateImuDynamics(X0, gyro, accel, dt, params->n_gravity);
  NavState X_inc = X0.compose(U);
  EXPECT(assert_equal(X_explicit, X_inc, 1e-12));
}

TEST(NavStateImuEKF, IncrementJacobianNumericalCheck) {
  using namespace nontrivial_navstate_example;
  double dt = 0.05;

  auto params = PreintegrationParams::MakeSharedU(9.81);
  NavStateImuEKF ekf(X0, I_9x9 * 1e-3, params);

  Matrix9 J;
  (void)navStateImuDynamics(X0, gyro, accel, dt, params->getGravity(), J);

  // Numerical Jacobian of u_left wrt X
  std::function<NavState(const NavState&)> ufun = [&](const NavState& X) {
    return navStateImuDynamics(X, gyro, accel, dt, params->getGravity());
  };
  Matrix9 Jnum = numericalDerivative11(ufun, X0);

  EXPECT(assert_equal(Jnum, J, 1e-6));
}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
