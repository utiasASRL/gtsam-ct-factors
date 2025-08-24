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

using namespace gtsam;

template <typename PoseType>
class Interpolator {
 protected:
  static constexpr int dim = traits<PoseType>::dimension;
  using VelocityType = typename gtsam::traits<PoseType>::TangentVector;
  using MatrixN = Eigen::Matrix<double, dim, dim>;
  using VectorN = Eigen::Matrix<double, dim, 1>;
  using Matrix2N = Eigen::Matrix<double, 2 * dim, 2 * dim>;
  using Vector2N = Eigen::Matrix<double, 2 * dim, 1>;
  using MatrixNx2N = Eigen::Matrix<double, dim, 2 * dim>;

  VectorN Q_psd_;  // Diagonal power Spectral Density for WNOA
  std::function<Matrix(double dt)> transitionFunction_;
  std::function<Matrix(double dt, const VectorN& Q_psd)> covarianceFunction_;
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
  using TimestampKeyMap = std::map<double, std::pair<Key, Key>>;

  // Maps a pose or velocity to their covariance matrix
  using CovarianceMap = std::map<Key, Matrix>;
  // Note (Daniel): bit iffy about using double for timestamps as
  // keys, but it should be fine as long as the keys are not modified afterwards

  Interpolator() = delete;

  Interpolator(
      const VectorN& Q_psd, std::function<Matrix(double dt)> transitionFunction,
      std::function<Matrix(double dt, const VectorN& Q_psd)> covarianceFunction,
      std::function<Matrix(const std::pair<PoseType, VelocityType>&,
                           const std::pair<PoseType, VelocityType>&, double)>
          computeJacobianPrev,
      std::function<Matrix(const std::pair<PoseType, VelocityType>&,
                           const std::pair<PoseType, VelocityType>&, double)>
          computeJacobianNext)
      : Q_psd_(Q_psd),
        transitionFunction_(transitionFunction),
        covarianceFunction_(covarianceFunction),
        computeJacobianPrev_(computeJacobianPrev),
        computeJacobianNext_(computeJacobianNext) {}

  // Default to WNOA
  Interpolator(const VectorN& Q_psd)
      : Interpolator(Q_psd, WNOAMotionFactor<PoseType>::transitionFunction,
                     WNOAMotionFactor<PoseType>::buildWNOACovariance,
                     WNOAMotionFactor<PoseType>::computeJacobianPrev,
                     WNOAMotionFactor<PoseType>::computeJacobianNext) {}

  std::pair<PoseType, VelocityType> interpolatePoseAndVelocity(
      std::pair<PoseType, VelocityType> Tvarpi_k, double t_k,
      std::pair<PoseType, VelocityType> Tvarpi_kp1, double t_kp1, double t_tau,
      OptionalMatrixVecType H = nullptr,
      const std::shared_ptr<Matrix>& mainSolveMarginalMatrix = nullptr,
      Matrix* covarianceOut = nullptr) const {
    // if t_tau is equal to t_k or t_kp1, return the corresponding pose and
    // velocity
    // NOTE: (CTH) need to make these first cases also provide jacobians and
    // covariances if necessary
    if (equal(t_tau, t_k)) {
      return Tvarpi_k;

    } else if (equal(t_tau, t_kp1)) {
      return Tvarpi_kp1;

    } else if (t_tau < t_k || t_tau > t_kp1 || std::isinf(t_k) ||
               std::isinf(t_kp1)) {
      auto Tvarpi_extrapolate_point =
          (t_tau < t_k || std::isinf(t_kp1)) ? Tvarpi_k : Tvarpi_kp1;
      double t_diff;
      if (t_tau < t_k || std::isinf(t_kp1)) {
        t_diff = t_tau - t_k;
      } else if (t_tau > t_kp1 || std::isinf(t_k)) {
        t_diff = t_tau - t_kp1;
      } else {  // shouldn't happen unless this code is bugged
        throw std::runtime_error(
            "Unexpected case in interpolatePoseAndVelocity");
      }
      // follow (11.5) in the book
      auto [T_ex, varpi_ex] = Tvarpi_extrapolate_point;
      Vector2N gamma_k;
      gamma_k.topRows(dim).setZero();
      gamma_k.bottomRows(dim) = varpi_ex;
      auto Psi = transitionFunction_(t_diff);
      auto gamma_ex = Psi * gamma_k;
      auto T_tau = traits<PoseType>::Compose(
          T_ex, traits<PoseType>::Expmap(gamma_ex.topRows(dim), nullptr));
      auto varpi_tau = gamma_ex.bottomRows(dim);

      if (mainSolveMarginalMatrix) {
        // compute covariance of the extrapolated pose and velocity
        // assume that mainSolveMarginalMatrix corresponds to the covariance of
        // Tvarpi_extrapolate_point
        assert(mainSolveMarginalMatrix->rows() == 2 * dim &&
               mainSolveMarginalMatrix->cols() == 2 * dim);
        Matrix2N Sigma = covarianceFunction_(t_diff, Q_psd_);
        // (11.5) in the book
        *covarianceOut =
            Sigma + Psi * *mainSolveMarginalMatrix * Psi.transpose();
      }
      return std::make_pair(T_tau, varpi_tau);

    } else {
      auto [T_k, varpi_k] = Tvarpi_k;
      auto [T_kp1, varpi_kp1] = Tvarpi_kp1;

      Matrix2N Lambda,
          Psi;  // ensure Lambda and Psi are 2*dim x 2*dim matrices,
                // rather than using auto in the line below
      std::tie(Lambda, Psi) = getLamdaPsi(t_k, t_kp1, t_tau);

      MatrixNx2N Lambda_1 = Lambda.topRows(dim);
      MatrixNx2N Lambda_2 = Lambda.bottomRows(dim);
      MatrixNx2N Psi_1 = Psi.topRows(dim);
      MatrixNx2N Psi_2 = Psi.bottomRows(dim);

      // form quantities for Eq. (11.45) in the book, (5.13) in the paper
      Vector2N gamma_k;
      gamma_k.topRows(dim).setZero();
      gamma_k.bottomRows(dim) = varpi_k;

      // Note that p1 = T(t_k), p2 = T(t_{k+1})
      //  compute xi = log(T_k^-1 T_{k+1})^check
      MatrixN dxi_dTk;
      MatrixN dxi_dTkp1;
      VectorN xi;
      MatrixN right_jac_inv;
      if (H) {
        MatrixN dbetween_Tk;
        MatrixN dbetween_Tkp1;
        xi = traits<PoseType>::Logmap(
            traits<PoseType>::Between(T_k, T_kp1, &dbetween_Tk, &dbetween_Tkp1),
            &right_jac_inv);
        dxi_dTk = right_jac_inv * dbetween_Tk;
        dxi_dTkp1 = right_jac_inv * dbetween_Tkp1;
      } else {
        xi = traits<PoseType>::Logmap(traits<PoseType>::Between(T_k, T_kp1),
                                      &right_jac_inv);
      }

      Vector2N gamma_kp1;
      gamma_kp1 << xi, right_jac_inv * varpi_kp1;

      auto xi_tau = Lambda_1 * gamma_k + Psi_1 * gamma_kp1;
      auto xidot_tau = Lambda_2 * gamma_k + Psi_2 * gamma_kp1;
      // Eq. (11.45) in Barfoot 2025
      MatrixN right_jac_tau;
      MatrixN dTtau_dTk;
      MatrixN dTtau_dxitau;
      PoseType T_tau;
      if (H) {
        T_tau = traits<PoseType>::Compose(
            T_k, traits<PoseType>::Expmap(xi_tau, &right_jac_tau), &dTtau_dTk,
            &dTtau_dxitau);
        dTtau_dxitau = dTtau_dxitau * right_jac_tau;
      } else {
        T_tau = traits<PoseType>::Compose(
            T_k, traits<PoseType>::Expmap(xi_tau, &right_jac_tau));
      }
      auto varpi_tau = right_jac_tau * xidot_tau;

      // Compute Jacobians
      if (H) {
        // Derivative of right Jacobians
        // Zero for vector spaces, use an approximation for Lie groups
        MatrixN dxidot_dxi;
        MatrixN dvarpitau_dxitau;
        if constexpr (std::is_same_v<
                          typename traits<PoseType>::structure_category,
                          vector_space_tag>) {
          dxidot_dxi.setZero();
          dvarpitau_dxitau.setZero();
        } else {
          // For Lie groups
          dxidot_dxi = -PoseType::adjointMap(varpi_kp1) / 2.0;
          dvarpitau_dxitau = PoseType::adjointMap(xidot_tau) / 2.0;
        }
        // dgammakp1
        Eigen::Matrix<double, 2 * dim, dim> dgammakp1_dTk;
        dgammakp1_dTk << dxi_dTk, dxidot_dxi * dxi_dTk;
        Eigen::Matrix<double, 2 * dim, dim> dgammakp1_dTkp1;
        dgammakp1_dTkp1 << dxi_dTkp1, dxidot_dxi * dxi_dTkp1;
        Eigen::Matrix<double, 2 * dim, dim> dgammakp1_dvarpikp1;
        dgammakp1_dvarpikp1 << Eigen::Matrix<double, dim, dim>::Zero(),
            right_jac_inv;
        // dxitau
        MatrixN dxitau_dTk = Psi_1 * dgammakp1_dTk;
        MatrixN dxitau_dvarpik = Lambda_1.template block<dim, dim>(0, dim);
        MatrixN dxitau_dTkp1 = Psi_1 * dgammakp1_dTkp1;
        MatrixN dxitau_dvarpikp1 = Psi_1 * dgammakp1_dvarpikp1;
        // dxidottau
        MatrixN dxidottau_dTk = Psi_2 * dgammakp1_dTk;
        MatrixN dxidottau_dvarpik = Lambda_2.template block<dim, dim>(0, dim);
        MatrixN dxidottau_dTkp1 = Psi_2 * dgammakp1_dTkp1;
        MatrixN dxidottau_dvarpikp1 = Psi_2 * dgammakp1_dvarpikp1;

        // dTtau_dTk
        (*H)[0] = dTtau_dTk + dTtau_dxitau * dxitau_dTk;
        // dTtau_dvarpik
        (*H)[1] = dTtau_dxitau * dxitau_dvarpik;
        // dTtau_dTkp1
        (*H)[2] = dTtau_dxitau * dxitau_dTkp1;
        // dTtau_dvarpikp1
        (*H)[3] = dTtau_dxitau * dxitau_dvarpikp1;
        // dvarpitau_dTk
        (*H)[4] = right_jac_tau * dxidottau_dTk + dvarpitau_dxitau * dxitau_dTk;
        // dvarpitau_dvarpik
        (*H)[5] = right_jac_tau * dxidottau_dvarpik +
                  dvarpitau_dxitau * dxitau_dvarpik;
        // dvarpitau_dTkp1
        (*H)[6] =
            right_jac_tau * dxidottau_dTkp1 + dvarpitau_dxitau * dxitau_dTkp1;
        // dvarpitau_dvarpikp1
        (*H)[7] = right_jac_tau * dxidottau_dvarpikp1 +
                  dvarpitau_dxitau * dxitau_dvarpikp1;
      }

      return std::make_pair(T_tau, varpi_tau);
    }
  }

  Values interpolatePosesAndVelocities(
      const NonlinearFactorGraph& mainSolveGraph,
      const Values& mainSolveSolution, const TimestampKeyMap& mainSolveKeyMap,
      const TimestampKeyMap& interpolateKeyMap,
      std::shared_ptr<CovarianceMap> covarianceMapOut = nullptr) const {
    // Map from intervals [t1, t2) to query times inside that interval (bucket)
    std::map<std::pair<double, double>, std::vector<double>> queryBuckets;

    for (const auto& [t_tau, keys] : interpolateKeyMap) {
      if (t_tau < mainSolveKeyMap.begin()->first) {
        queryBuckets[std::make_pair(-std::numeric_limits<double>::infinity(),
                                    mainSolveKeyMap.begin()->first)]
            .push_back(t_tau);
      } else if (t_tau > mainSolveKeyMap.rbegin()->first) {
        queryBuckets[std::make_pair(mainSolveKeyMap.rbegin()->first,
                                    std::numeric_limits<double>::infinity())]
            .push_back(t_tau);
      } else {
        auto it2 = mainSolveKeyMap.upper_bound(t_tau);
        auto it1 = std::prev(it2);
        queryBuckets[std::make_pair(it1->first, it2->first)].push_back(t_tau);
      }
    }

    Values interpolatedSolution;

    std::unique_ptr<Marginals>
        marginals;  // Only construct a Marginals if requested
    if (covarianceMapOut) {
      marginals =
          std::make_unique<Marginals>(mainSolveGraph, mainSolveSolution);
    }
    for (const auto& [interval, times] : queryBuckets) {
      double t_k = interval.first;
      double t_kp1 = interval.second;

      // Get the poses and velocities at t_k and t_kp1
      auto pvk = std::isinf(t_k)
                     ? std::make_pair(PoseType(), VelocityType())
                     : std::make_pair(mainSolveSolution.at<PoseType>(
                                          mainSolveKeyMap.at(t_k).first),
                                      mainSolveSolution.at<VelocityType>(
                                          mainSolveKeyMap.at(t_k).second));
      auto pvkp1 = std::isinf(t_kp1)
                       ? std::make_pair(PoseType(), VelocityType())
                       : std::make_pair(mainSolveSolution.at<PoseType>(
                                            mainSolveKeyMap.at(t_kp1).first),
                                        mainSolveSolution.at<VelocityType>(
                                            mainSolveKeyMap.at(t_kp1).second));

      // Compute covariances of the interpolated poses and velocities
      std::shared_ptr<Matrix> mainSolveMarginalMatrix;  // (4*dim, 4*dim)
      Matrix covarianceOut;                             // (2*dim, 2*dim)
      if (covarianceMapOut) {
        // following (5.22) in paper
        KeyVector variables;
        if (std::isinf(t_k)) {
          variables = {mainSolveKeyMap.at(t_kp1).first,
                       mainSolveKeyMap.at(t_kp1).second};  // {p2, v2}
        } else if (std::isinf(t_kp1)) {
          variables = {mainSolveKeyMap.at(t_k).first,
                       mainSolveKeyMap.at(t_k).second};  // {p1, v1}
        } else {
          variables = {
              // {p1, v1, p2, v2}
              mainSolveKeyMap.at(t_k).first, mainSolveKeyMap.at(t_k).second,
              mainSolveKeyMap.at(t_kp1).first,
              mainSolveKeyMap.at(t_kp1).second};
        }
        JointMarginal mainSolveMarginal =
            marginals->jointMarginalCovariance(variables);
        // avoid using JointMarginal.fullMatrix() as it returns covariance
        // in alphabetical order of the keys...
        mainSolveMarginalMatrix =
            std::make_shared<Matrix>(constructMatrixFromJointMarginal(
                mainSolveMarginal, variables, dim));
      }

      // Interpolate for all query times within this query interval (bucket)
      for (double t_tau : times) {
        auto pvtau =
            interpolatePoseAndVelocity(pvk, t_k, pvkp1, t_kp1, t_tau, nullptr,
                                       mainSolveMarginalMatrix, &covarianceOut);
        auto [T_tau, varpi_tau] = pvtau;
        interpolatedSolution.insert(interpolateKeyMap.at(t_tau).first, T_tau);
        interpolatedSolution.insert(interpolateKeyMap.at(t_tau).second,
                                    varpi_tau);
        if (covarianceMapOut) {
          // upper left covariance block corresponds to pose, lower right block
          // corresponds to velocity
          (*covarianceMapOut)[interpolateKeyMap.at(t_tau).first] =
              covarianceOut.topLeftCorner(dim, dim);
          (*covarianceMapOut)[interpolateKeyMap.at(t_tau).second] =
              covarianceOut.bottomRightCorner(dim, dim);
        }
      }
    }
    return interpolatedSolution;
  }

  std::pair<Matrix, Matrix> getLamdaPsi(double t_k, double t_kp1,
                                        double t_tau) const {
    // Build transition matrices for all combinations
    double dt = t_kp1 - t_k;
    auto Phi_12 = transitionFunction_(dt);
    auto Phi_1tau = transitionFunction_(t_tau - t_k);
    auto Phi_tau2 = transitionFunction_(t_kp1 - t_tau);

    // Construct Q
    auto Q_12 = covarianceFunction_(dt, Q_psd_);
    auto Q_1tau = covarianceFunction_(t_tau - t_k, Q_psd_);

    // Eq. (11.41) in the book
    auto Lambda =
        Phi_1tau - Q_1tau * Phi_tau2.transpose() * Q_12.inverse() * Phi_12;
    auto Psi = Q_1tau * Phi_tau2.transpose() * Q_12.inverse();

    return std::make_pair(Lambda, Psi);
  }

  Matrix2N computeConditionalCov(const std::pair<PoseType, VelocityType>& pvk,
                                 const std::pair<PoseType, VelocityType>& pvkp1,
                                 const std::pair<PoseType, VelocityType>& pvtau,
                                 double t_k, double t_kp1, double t_tau) const {
    // see Figure 5.4 in the paper
    Matrix2N Q_tau_prev = covarianceFunction_(t_tau - t_k, Q_psd_);
    Matrix2N Q_tau_next = covarianceFunction_(t_kp1 - t_tau, Q_psd_);
    Matrix2N E_tau = computeJacobianPrev_(pvk, pvtau, t_tau - t_k);
    Matrix2N F_tau = computeJacobianNext_(pvtau, pvkp1, t_kp1 - t_tau);
    Matrix2N Sigma_inv = E_tau.transpose() * Q_tau_prev.inverse() * E_tau +
                         F_tau.transpose() * Q_tau_next.inverse() * F_tau;
    Matrix2N Sigma = Sigma_inv.inverse();
    return Sigma;
  }

 protected:
  static Matrix constructMatrixFromJointMarginal(
      // construct the full covariance matrix using the order given by keyVector
      const JointMarginal& blockMatrix, const KeyVector& keyVector,
      size_t blockSize) {
    size_t n = keyVector.size();
    Matrix M(n * blockSize, n * blockSize);

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j <= i;
           ++j) {  // symmetric block matrix: j <= i to avoid repeats
        auto block = blockMatrix(keyVector[i], keyVector[j]);  // query block
        // fill in both (i, j) and (j, i) blocks
        M.block(i * blockSize, j * blockSize, blockSize, blockSize) = block;
        if (i != j) {
          M.block(j * blockSize, i * blockSize, blockSize, blockSize) =
              block.transpose();
        }
      }
    }
    return M;
  }

  // Not used anymore, but may be useful in future
  static Matrix reorderSymmetricMatrix(const Matrix& mat, size_t block_size,
                                       const std::vector<size_t>& block_order) {
    // This was previously used to reorder a matrix from
    // JointMarginal.fullMatrix() e.g. if KeyVector is {p1, v1, p2, v2},
    // JointMarginal.fullMatrix() would be the marginal corresponding to {v1,
    // v2, p1, p2}. Then, we could call
    // reorderSymmetricMatrix(mainSolveMarginalPermuted, dim, {2, 0, 3, 1});
    assert(mat.rows() == mat.cols() && "Matrix must be square");
    assert(block_order.size() * block_size == static_cast<size_t>(mat.rows()) &&
           "Block order size must match matrix dimensions");
    Matrix reordered(mat.rows(), mat.cols());
    for (size_t i = 0; i < block_order.size(); ++i)
      for (size_t j = 0; j < block_order.size(); ++j)
        reordered.block(i * block_size, j * block_size, block_size,
                        block_size) =
            mat.block(block_order[i] * block_size, block_order[j] * block_size,
                      block_size, block_size);

    return reordered;
  }
};

// Explicit Instantiation.
template class Interpolator<Point1>;
template class Interpolator<Point2>;
template class Interpolator<Point3>;
template class Interpolator<Pose2>;
template class Interpolator<Pose3>;