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
 * Free dynamics function for IMU-driven NavState on SE_2(3).
 * Computes the left-trivialized body-frame tangent xi = [omega, R^T v, a +
 * R^T n_gravity]. H (if provided) is d(xi)/d(local(X)) with local(X) ordered
 * as [dR,dP,dV].
 *
 * @param X Current NavState.
 * @param gyro Body angular velocity measurement (rad/s).
 * @param accel Body specific force measurement (m/s^2).
 * @param n_gravity Gravity vector expressed in the navigation frame.
 * @param H Optional Jacobian d(xi)/d(local(X)).
 * @return Tangent vector xi in order [dR,dP,dV].
 */
Vector9 GTSAM_EXPORT navStateImuDynamics(const NavState& X, const Vector3& gyro,
                                         const Vector3& accel,
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

  /// Predict with gyro and accel controls; uses Base::predict with
  /// state-dependent dynamics.
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