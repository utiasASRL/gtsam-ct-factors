/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file EqVIOSymmetry.h
 * @brief EqVIO symmetry actions and lift helpers.
 * @author Rohan Bansal
 */

#pragma once

#include <gtsam/base/GroupAction.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam_unstable/navigation/EqVIOCommon.h>
#include <gtsam_unstable/navigation/EqVIOState.h>
#include <gtsam_unstable/dllexport.h>

namespace gtsam {
namespace eqvio {

/**
 * @brief Right action of `VioGroup` on the sensor-only state block.
 *
 * Applies bias, body pose/velocity, and camera extrinsic transforms while
 * preserving the EqVIO right-action convention.
 */
GTSAM_UNSTABLE_EXPORT SensorState sensorStateGroupAction(
    const VioGroup& X, const SensorState& sensor);
/**
 * @brief Right action of `VioGroup` on full EqVIO state.
 *
 * Applies `sensorStateGroupAction` to the sensor block and SOT3 inverse action
 * to each landmark state.
 */
GTSAM_UNSTABLE_EXPORT State stateGroupAction(const VioGroup& X,
                                                const State& state);

/**
 * @brief Discrete lift map from IMU input to a `VioGroup` increment.
 *
 * This is the discrete propagation primitive used by `EqVIOFilter::propagateState`.
 */
GTSAM_UNSTABLE_EXPORT VioGroup liftVelocityDiscrete(const State& state,
                                                    const IMUInput& velocity,
                                                    double dt);

/**
 * @brief Generate ideal image measurements by projecting all landmarks.
 * @throws std::invalid_argument if `camera` is null.
 */
GTSAM_UNSTABLE_EXPORT VisionMeasurement measureSystemState(
    const State& state, const std::shared_ptr<const CameraModel>& camera);

/**
 * @brief EqF linearization blocks used by EqVIO.
 *
 * EqVIO uses an inverse-depth landmark error chart to preserve the
 * equivariant output linearization while remaining numerically stable for
 * distant points. Other coordinate systems (Euclid, Polar) are not supported
 * in this implementation yet. See the EqVIO paper (van Goor and Mahony,
 * arXiv:2205.01980v3).
 */
GTSAM_UNSTABLE_EXPORT Matrix EqFStateMatrixA(
    const VioGroup& X, const State& xi0, const IMUInput& imuVel);
/// EqF input matrix that maps IMU driving noise into chart error coordinates.
GTSAM_UNSTABLE_EXPORT Matrix EqFInputMatrixB(
    const VioGroup& X, const State& xi0);
/// Per-landmark equivariant output Jacobian using observed measurement `y`.
GTSAM_UNSTABLE_EXPORT Matrix23 EqFoutputMatrixCiStar(
    const Point3& q0, const SOT3& QHat,
    const std::shared_ptr<const CameraModel>& camera, const Point2& y);
/// Per-landmark output Jacobian using predicted measurement from current state.
GTSAM_UNSTABLE_EXPORT Matrix23 EqFoutputMatrixCi(
    const Point3& q0, const SOT3& QHat,
    const std::shared_ptr<const CameraModel>& camera);
/// Stacked output matrix for all currently observed landmarks.
GTSAM_UNSTABLE_EXPORT Matrix EqFoutputMatrixC(
    const State& xi0, const VioGroup& X, const VisionMeasurement& y,
    const std::shared_ptr<const CameraModel>& camera,
    bool useEquivariance = true);
/// Map state innovation in chart coordinates to group tangent innovation.
GTSAM_UNSTABLE_EXPORT Vector liftInnovation(const Vector& totalInnovation,
                                            const State& xi0);

/// Right action phi(xi, X) = stateGroupAction(X, xi).
struct GTSAM_UNSTABLE_EXPORT Symmetry
    : public GroupAction<Symmetry, VioGroup, State> {
  static constexpr ActionType type = ActionType::Right;

  /**
   * @brief Evaluate right action `phi(xi, X)`.
   * @param xi State argument.
   * @param X Group argument.
   * @param H_xi Optional Jacobian w.r.t. state argument.
   * @param H_X Optional Jacobian w.r.t. group argument.
   */
  State operator()(const State& xi, const VioGroup& X,
                      OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H_xi = {},
                      OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic> H_X = {})
      const;
};

}  // namespace eqvio
}  // namespace gtsam
