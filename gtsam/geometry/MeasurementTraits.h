/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file MeasurementTraits.h
 *
 * @brief Traits for computing measurement errors with proper Jacobians
 *
 * This file provides a helper for computing the error between a measured
 * and predicted value, along with the Jacobian. For most measurement types,
 * this is simply the difference (using traits<T>::Local). However, Unit3
 * requires special handling via errorVector to get correct Jacobians.
 *
 * @date 2024
 */

#pragma once

#include <gtsam/base/Manifold.h>
#include <gtsam/base/OptionalJacobian.h>
#include <gtsam/geometry/Unit3.h>

namespace gtsam {
namespace internal {

/**
 * @brief Helper for computing measurement error with Jacobian
 *
 * For general measurement types, computes error = Local(measured, predicted)
 * with identity Jacobian (since Local is the canonical local coordinates).
 *
 * Specialized for Unit3 to use errorVector which provides correct Jacobians.
 */
template <class MEASUREMENT>
struct MeasurementErrorHelper {
  static constexpr int Dim = traits<MEASUREMENT>::dimension;
  using VectorType = Eigen::Matrix<double, Dim, 1>;
  using MatrixType = Eigen::Matrix<double, Dim, Dim>;

  /**
   * Compute the error between measured and predicted values
   * @param measured The measured value
   * @param predicted The predicted value
   * @param H_predicted Optional Jacobian w.r.t. predicted
   * @return Error vector in local coordinates
   */
  static VectorType Evaluate(const MEASUREMENT& measured,
                             const MEASUREMENT& predicted,
                             OptionalJacobian<Dim, Dim> H_predicted) {
    if (H_predicted) *H_predicted = MatrixType::Identity();
    auto local = traits<MEASUREMENT>::Local(measured, predicted);
    return VectorType(local);
  }
};

/**
 * @brief Specialization for Unit3 measurements
 *
 * Unit3 requires errorVector instead of Local to get correct Jacobians.
 */
template <>
struct MeasurementErrorHelper<Unit3> {
  static constexpr int Dim = traits<Unit3>::dimension;
  using VectorType = Eigen::Matrix<double, Dim, 1>;

  static VectorType Evaluate(const Unit3& measured, const Unit3& predicted,
                             OptionalJacobian<Dim, Dim> H_predicted) {
    return measured.errorVector(predicted, {}, H_predicted);
  }
};

}  // namespace internal
}  // namespace gtsam
