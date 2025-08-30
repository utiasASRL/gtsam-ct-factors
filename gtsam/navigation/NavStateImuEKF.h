/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file  NavStateImuEKF.h
 * @brief Extended Kalman Filter for IMU-driven NavState on SE(3).
 *
 * @date  August 2025
 * @authors Derek Benham, Frank Dellaert
 */

#pragma once

#include <gtsam/navigation/LieGroupEKF.h>  // Include the base class
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/PreintegrationParams.h>

namespace gtsam {

/**
 * IMU-driven NavState on SE_2(3).
 * Returns a group increment U(X) that realizes a kinematic, second-order
 * integration step which cannot be reproduced by the Euler tangent-based
 * step in LieGroupEKF. Specifically, for X = (R,p,v):
 *   dR = Exp(gyro * dt)
 *   dv_body = (accel + R^T n_gravity) * dt
 *   dp_body = (R^T v) * dt + 0.5 * (accel + R^T n_gravity) * dt^2
 * and U = (dR, dp_body, dv_body). Composition X+ = X * U then yields exactly:
 *   R+ = R dR,
 *   v+ = v + (R accel + n_gravity) dt,
 *   p+ = p + v dt + (R accel + n_gravity) 0.5 dt^2.
 *
 * Note that this implements a custom integration and is intentionally
 * different from the Euler update used by LieGroupEKF::predict with a tangent
 * dynamics functor.
 */
GTSAM_EXPORT NavState navStateImuDynamics(const NavState& X,
                                          const Vector3& gyro,
                                          const Vector3& accel, double dt,
                                          const Vector3& n_gravity,
                                          OptionalJacobian<9, 9> H = {});

/// Specialized EKF for IMU-driven NavState on SE_2(3)
class GTSAM_EXPORT NavStateImuEKF : public LieGroupEKF<NavState> {
 public:
  using Base = LieGroupEKF<NavState>;
  using TangentVector = typename Base::TangentVector;  // Vector9
  using Jacobian = typename Base::Jacobian;            // 9x9
  using Covariance = typename Base::Covariance;        // 9x9

  /// Construct with initial state/covariance and preintegration params (for
  /// gravity and IMU covariances)
  /// @param X0 Initial NavState.
  /// @param P0 Initial covariance in tangent space at X0.
  /// @param params Preintegration parameters providing gravity and options.
  NavStateImuEKF(const NavState& X0, const Covariance& P0,
                 const std::shared_ptr<PreintegrationParams>& params);

  /// Predict with gyro and accel using the custom increment integrator above.
  /// @param gyro Body angular velocity measurement (rad/s).
  /// @param accel Body specific force measurement (m/s^2).
  /// @param dt Time step in seconds.
  void predict(const Vector3& gyro, const Vector3& accel, double dt);

  /// Accessors
  const std::shared_ptr<PreintegrationParams>& params() const;
  const Vector3& gravity() const;
  const Covariance& processNoise() const;

 private:
  std::shared_ptr<PreintegrationParams> params_;
  Covariance Q_ = Covariance::Zero();
};

}  // namespace gtsam