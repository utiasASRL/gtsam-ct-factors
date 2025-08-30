/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file  NavStateImuEKF.cpp
 * @brief Extended Kalman Filter derived class for IMU-driven NavState.
 *
 * @date  August 2025
 * @authors Derek Benham, Frank Dellaert
 */

#include <gtsam/base/Matrix.h>
#include <gtsam/navigation/NavStateImuEKF.h>

namespace gtsam {

Vector9 navStateImuDynamics(const NavState& X, const Vector3& gyro,
                            const Vector3& accel, const Vector3& n_gravity,
                            OptionalJacobian<9, 9> H) {
  Vector9 xi;
  xi.setZero();
  // Rotation, position, velocity in NavState
  const Rot3& R = X.attitude();
  const Vector3& v = X.velocity();

  // Body-frame quantities needed for left-trivialized tangent
  const Vector3 v_body = R.unrotate(v);          // R^T v
  const Vector3 g_body = R.unrotate(n_gravity);  // R^T g

  // Tangent vector components (dR,dP,dV)
  NavState::dR(xi) = gyro;            // omega (body)
  NavState::dP(xi) = v_body;          // p_dot in body frame
  NavState::dV(xi) = accel + g_body;  // v_dot in body frame

  if (H) {
    H->setZero();
    // xi_rot = gyro -> no dependence on state
    // xi_trans = R^T v
    // d(xi_trans)/d(dR) = +skew(R^T v) ; d(xi_trans)/d(dV) = I ; d/d(dP) = 0
    H->template block<3, 3>(3, 0) = skewSymmetric(v_body);
    H->template block<3, 3>(3, 6) = I_3x3;
    // xi_vel = a + R^T g
    // d(xi_vel)/d(dR) = +skew(R^T g)
    H->template block<3, 3>(6, 0) = skewSymmetric(g_body);
  }

  return xi;
}

// ===== NavStateImuEKF methods =====

NavStateImuEKF::NavStateImuEKF(
    const NavState& X0, const Covariance& P0,
    const std::shared_ptr<PreintegrationParams>& params)
    : Base(X0, P0), params_(params) {
  // Build process noise Q_ = block_diag(Cg, Ci, Ca)
  const Matrix3& Cg = params_->gyroscopeCovariance;
  const Matrix3& Ci = params_->integrationCovariance;
  const Matrix3& Ca = params_->accelerometerCovariance;
  Q_.setZero();
  Q_.template block<3, 3>(0, 0) = Cg;
  Q_.template block<3, 3>(3, 3) = Ci;
  Q_.template block<3, 3>(6, 6) = Ca;
}

void NavStateImuEKF::predict(const Vector3& gyro, const Vector3& accel,
                             double dt) {
  auto dyn = [&](const NavState& X, OptionalJacobian<Dim, Dim> H) {
    return navStateImuDynamics(X, gyro, accel, params_->n_gravity, H);
  };
  Base::predict(dyn, dt, Q_);
}

const std::shared_ptr<PreintegrationParams>& NavStateImuEKF::params() const {
  return params_;
}

const Vector3& NavStateImuEKF::gravity() const { return params_->n_gravity; }

const NavStateImuEKF::Covariance& NavStateImuEKF::processNoise() const {
  return Q_;
}

}  // namespace gtsam