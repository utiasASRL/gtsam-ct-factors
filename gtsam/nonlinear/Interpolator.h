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
#include <gtsam/nonlinear/StateData.h>
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
#include <unordered_set>

namespace gtsam {

// Intervals of boundary states
// empty optional = unbounded
using StateDataInterval = std::pair<std::optional<StateData>, std::optional<StateData>>;


template <typename PoseType>
struct PoseVelocity {
  PoseType pose;
  typename traits<PoseType>::TangentVector vel;

  std::pair<PoseType, typename traits<PoseType>::TangentVector> asPair() const {
    return std::make_pair(pose, vel);
  }
};

template <typename PoseType>
struct TimestampedPoseVelocity {
  PoseVelocity<PoseType> poseVel;
  double timestamp;

  TimestampedPoseVelocity(PoseType pose,
                          typename traits<PoseType>::TangentVector vel,
                          double time)
      : poseVel{pose, vel}, timestamp(time) {}

  TimestampedPoseVelocity(PoseVelocity<PoseType> pv, double time)
      : poseVel(pv), timestamp(time) {}
};

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
  std::function<Matrix(double dt, const VectorN& Q_psd)>
      inverseCovarianceFunction_;
  // Todo: need to make the below two functions generalize to cases with no
  // velocities, e.g. WNOV
  std::function<Matrix(const std::pair<PoseType, VelocityType>&,
                       const std::pair<PoseType, VelocityType>&, double)>
      computeJacobianPrev_;
  std::function<Matrix(const std::pair<PoseType, VelocityType>&,
                       const std::pair<PoseType, VelocityType>&, double)>
      computeJacobianNext_;

 public:
  using StateDataSet = std::set<StateData>;
  using PoseVel = PoseVelocity<PoseType>;
  using TimestampedPoseVel = TimestampedPoseVelocity<PoseType>;
  using LambdaPsiMats = std::pair<Matrix2N, Matrix2N>;

  // Maps a pose or velocity to their covariance matrix
  using CovarianceMap = std::map<Key, Matrix>;

  Interpolator() = delete;

  Interpolator(
      const VectorN& Q_psd, std::function<Matrix(double dt)> transitionFunction,
      std::function<Matrix(double dt, const VectorN& Q_psd)> covarianceFunction,
      std::function<Matrix(double dt, const VectorN& Q_psd)>
          inverseCovarianceFunction,
      std::function<Matrix(const std::pair<PoseType, VelocityType>&,
                           const std::pair<PoseType, VelocityType>&, double)>
          computeJacobianPrev,
      std::function<Matrix(const std::pair<PoseType, VelocityType>&,
                           const std::pair<PoseType, VelocityType>&, double)>
          computeJacobianNext);

  // Default to WNOA
  Interpolator(const VectorN& Q_psd);

  PoseVel interpolatePoseAndVelocity(
      const std::optional<TimestampedPoseVel>& Tvarpi_k, const std::optional<TimestampedPoseVel>& Tvarpi_kp1,
      double t_tau, OptionalMatrixVecType H = nullptr,
      const std::shared_ptr<Matrix>& mainSolveMarginalMatrix = nullptr,
      Matrix* covarianceOut = nullptr,
      const std::shared_ptr<const LambdaPsiMats>& LambdaPsiPreComp = nullptr) const;

  Values interpolatePosesAndVelocities(
      const NonlinearFactorGraph& mainSolveGraph,
      const Values& mainSolveSolution, const StateDataSet& mainSolveStates,
      const StateDataSet& interpolatedStates,
      std::shared_ptr<CovarianceMap> covarianceMapOut = nullptr) const;

  // if Lambda and Psi are provided, they will be computed using (5.23)
  Matrix2N computeConditionalCov(const TimestampedPoseVel& pvk,
                                 const TimestampedPoseVel& pvkp1,
                                 const TimestampedPoseVel& pvtau,
                                 OptionalMatrixType Lambda = nullptr,
                                 OptionalMatrixType Psi = nullptr) const;

  // Retrieve interpolation matrices. Fast implementation specialized for WNOA.
  std::pair<Matrix, Matrix> getLambdaPsi(double t_k, double t_kp1,
                                         double t_tau) const;

  // Retrieve interpolation matrices. General implementation.
  std::pair<Matrix, Matrix> getLambdaPsiGeneral(double t_k, double t_kp1,
                                                double t_tau) const;


 protected:
  // Interpoate pose and velocity at left boundary
  PoseVel interpolateBoundaryLeft(
      const PoseVelocity<PoseType>& poseVel_k,
      OptionalMatrixVecType H = nullptr,
      const std::shared_ptr<Matrix>& mainSolveMarginalMatrix = nullptr,
      Matrix* covarianceOut = nullptr) const;

  // Interpoate pose and velocity at right boundary
  PoseVel interpolateBoundaryRight(
      const PoseVelocity<PoseType>& poseVel_kp1,
      OptionalMatrixVecType H = nullptr,
      const std::shared_ptr<Matrix>& mainSolveMarginalMatrix = nullptr,
      Matrix* covarianceOut = nullptr) const;

  // Extrapolation case for pose and velocity
  PoseVel extrapolatePoseAndVelocity(
      const PoseVelocity<PoseType>& poseVel, double t_diff,
      OptionalMatrixVecType H = nullptr,
      const std::shared_ptr<Matrix>& mainSolveMarginalMatrix = nullptr,
      Matrix* covarianceOut = nullptr) const;

  // Interpolate pose and velocity, Internal overload that actually does the
  // work
  PoseVel interpolatePoseAndVelocity_(
      const TimestampedPoseVel& tPoseVel_k,
      const TimestampedPoseVel& tPoseVel_kp1, double t_tau,
      OptionalMatrixVecType H = nullptr,
      const std::shared_ptr<Matrix>& mainSolveMarginalMatrix = nullptr,
      Matrix* covarianceOut = nullptr,
      const std::shared_ptr<const LambdaPsiMats>& LambdaPsiPreComp = nullptr) const;
  static std::map<StateDataInterval, std::shared_ptr<Matrix>>
  computeJointMarginals(
    const std::map<StateDataInterval, std::vector<StateData>>& queryBuckets,
    const std::unique_ptr<Marginals>& marginals);

  // construct the full covariance matrix using the order given by keyVector
  static Matrix constructMatrixFromJointMarginal(
      const JointMarginal& blockMatrix, const KeyVector& keyVector,
      size_t blockSize);

  // Not used anymore, but may be useful in future
  static Matrix reorderSymmetricMatrix(const Matrix& mat, size_t block_size,
                                       const std::vector<size_t>& block_order);
};

}  // namespace gtsam
