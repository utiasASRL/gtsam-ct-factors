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
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/navigation/NavStateImuEKF.h>

namespace gtsam {

NavState navStateImuDynamics(const NavState& X, const Vector3& gyro,
                             const Vector3& accel, double dt,
                             const Vector3& n_gravity,
                             OptionalJacobian<9, 9> H) {
  // Rotation and velocity
  const Rot3& R = X.attitude();
  Matrix3 D_vb_R, D_vb_v, D_gb_R;
  const Vector3& v_n = X.velocity();  // Has D_v_v = R !
  const Vector3 v_body =
      R.unrotate(v_n, H ? &D_vb_R : nullptr, H ? &D_vb_v : nullptr);
  const Vector3 g_body = R.unrotate(n_gravity, H ? &D_gb_R : nullptr);
  const Vector3 a_b_total = accel + g_body;

  // Construct increment directly as group element with body-frame p/v
  // increments
  const Rot3 dR = Rot3::Expmap(gyro * dt);
  const double dt2 = 0.5 * dt * dt;
  const Vector3 dp_body = v_body * dt + a_b_total * dt2;
  const Vector3 dv_body = a_b_total * dt;
  NavState U(dR, dp_body, dv_body);

  if (H) {
    Matrix3 dRt = dR.transpose();  // Jacobian of NavState::Create
    H->setZero();
    // position:
    H->template block<3, 3>(3, 0) = dRt * (D_vb_R * dt + D_gb_R * dt2);
    const Matrix3 D_v_v = R.matrix();  // Jacobian of velocity()
    H->template block<3, 3>(3, 6) = dRt * D_vb_v * D_v_v * dt;
    // velocity:
    H->template block<3, 3>(6, 0) = dRt * D_gb_R * dt;
  }

  return U;
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
  if (dt <= 0.0) {
    throw std::invalid_argument("NavStateImuEKF::predict: dt must be positive");
  }
  // Use the custom increment integrator; compute U and J at current state
  Jacobian J_UX;
  const NavState U = navStateImuDynamics(this->state(), gyro, accel, dt,
                                         params_->n_gravity, J_UX);
  // Scale continuous-time process noise to the discrete interval [t, t+dt]
  Covariance Qdt = Q_ * dt;  // simple dt scaling per block (gyro, integ, accel)
  // More sophisticated discretization can be done here:
  // With Qc = diag(Cg, Ca), take G = [[dt·I, 0], [ 0, 0.5·dt^2·I], [ 0, dt·I]]
  // and Qd = G Qc G^T. This yields correct dt, dt^2, dt^3 and p–v cross terms.
  Base::predictWithCompose(U, J_UX, Qdt);
}

const std::shared_ptr<PreintegrationParams>& NavStateImuEKF::params() const {
  return params_;
}

const Vector3& NavStateImuEKF::gravity() const { return params_->n_gravity; }

const NavStateImuEKF::Covariance& NavStateImuEKF::processNoise() const {
  return Q_;
}

}  // namespace gtsam