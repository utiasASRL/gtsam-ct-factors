/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file Interpolator.h
 * @brief For post-solve interpolation of poses and velocities
 * @date June 20, 2025
 * @author Zi Cong Guo
 */
#pragma once
// Both relative poses and recovered trajectory poses will be stored as Pose2.
#include <gtsam/geometry/Pose3.h>
using gtsam::Pose2;

// gtsam::Vectors are dynamic Eigen vectors, Vector3 is statically sized.
#include <gtsam/base/Vector.h>
using gtsam::Vector;
using gtsam::Vector2;
using gtsam::Vector3;

// Unknown landmarks are of type Point2 (which is just a 2D Eigen vector).
#include <gtsam/geometry/Point2.h>
using gtsam::Point2;

// Each variable in the system (poses and landmarks) must be identified with a
// unique key. We can either use simple integer keys (1, 2, 3, ...) or symbols
// (X1, X2, L1). Here we will use Symbols.
#include <gtsam/inference/Symbol.h>
using gtsam::Symbol;

// We want to use iSAM2 to solve the range-SLAM problem incrementally.
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Marginals.h>

// Measurement functions are represented as 'factors'. Several common factors
// have been provided with the library for solving robotics SLAM problems:
#include <gtsam/nonlinear/WNOAFactor.h>

using gtsam::tictoc_print_;

#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/slam/StereoFactor.h>

// Standard headers, added last, so we know headers above work on their own.
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

// Define a custom comparator for double keys to handle floating-point precision issues
struct DoubleCompare {
    bool operator()(double a, double b) const {
        return (a + 1e-9) < b; // define "equal" within a tolerance
    }
};

namespace gtsam {

template <typename PoseType>
class Interpolator {
 protected:
  static constexpr int dim = traits<PoseType>::dimension;
  using VelocityType = typename traits<PoseType>::TangentVector;
  using MatrixN = Eigen::Matrix<double, dim, dim>;
  using VectorN = Eigen::Matrix<double, dim, 1>;
  using Matrix2N = Eigen::Matrix<double, 2 * dim, 2 * dim>;
  using Vector2N = Eigen::Matrix<double, 2 * dim, 1>;
  using MatrixNx2N = Eigen::Matrix<double, dim, 2 * dim>;

  VectorN Q_psd_;  // Diagonal power Spectral Density for WNOA
  std::function<Matrix(double dt)> transitionFunction_;
  std::function<Matrix(double dt, const VectorN& Q_psd)> covarianceFunction_;
  std::function<Matrix(double dt, const VectorN& Q_psd)> inverseCovarianceFunction_;
  // Todo: need to make the below two functions generalize to cases with no
  // velocities, e.g. WNOV
  std::function<Matrix(const std::pair<PoseType, VelocityType>&,
                       const std::pair<PoseType, VelocityType>&, double)>
      computeJacobianPrev_;
  std::function<Matrix(const std::pair<PoseType, VelocityType>&,
                       const std::pair<PoseType, VelocityType>&, double)>
      computeJacobianNext_;

 public:
  // Maps timestamp to a pair of keys (pose, velocity)
  using TimestampKeyMap = std::map<double, std::pair<Key, Key>, DoubleCompare>;

  // Maps a pose or velocity to their covariance matrix
  using CovarianceMap = std::map<Key, Matrix>;
  // Note (Daniel): bit iffy about using double for timestamps as
  // keys, but it should be fine as long as the keys are not modified afterwards

  Interpolator() = delete;

  Interpolator(
      const VectorN& Q_psd, std::function<Matrix(double dt)> transitionFunction,
      std::function<Matrix(double dt, const VectorN& Q_psd)> covarianceFunction,
      std::function<Matrix(double dt, const VectorN& Q_psd)> inverseCovarianceFunction,
      std::function<Matrix(const std::pair<PoseType, VelocityType>&,
                           const std::pair<PoseType, VelocityType>&, double)>
          computeJacobianPrev,
      std::function<Matrix(const std::pair<PoseType, VelocityType>&,
                           const std::pair<PoseType, VelocityType>&, double)>
          computeJacobianNext);

  // Default to WNOA
  Interpolator(const VectorN& Q_psd);

  std::pair<PoseType, VelocityType> interpolatePoseAndVelocity(
      std::pair<PoseType, VelocityType> Tvarpi_k, double t_k,
      std::pair<PoseType, VelocityType> Tvarpi_kp1, double t_kp1, double t_tau,
      OptionalMatrixVecType H = nullptr,
      const std::shared_ptr<Matrix>& mainSolveMarginalMatrix = nullptr,
      Matrix* covarianceOut = nullptr) const;

  Values interpolatePosesAndVelocities(
      const NonlinearFactorGraph& mainSolveGraph,
      const Values& mainSolveSolution, const TimestampKeyMap& mainSolveKeyMap,
      const TimestampKeyMap& interpolateKeyMap,
      std::shared_ptr<CovarianceMap> covarianceMapOut = nullptr) const;

  std::pair<Matrix, Matrix> getLamdaPsi(double t_k, double t_kp1,
                                        double t_tau) const;

  Matrix2N computeConditionalCov(const std::pair<PoseType, VelocityType>& pvk,
                                 const std::pair<PoseType, VelocityType>& pvkp1,
                                 const std::pair<PoseType, VelocityType>& pvtau,
                                 double t_k, double t_kp1, double t_tau) const;

 protected:
  // construct the full covariance matrix using the order given by keyVector
  static Matrix constructMatrixFromJointMarginal(const JointMarginal& blockMatrix,
                                                 const KeyVector& keyVector,
                                                 size_t blockSize);

  // Not used anymore, but may be useful in future
  static Matrix reorderSymmetricMatrix(const Matrix& mat, size_t block_size,
                                       const std::vector<size_t>& block_order);
};

} // namespace gtsam
