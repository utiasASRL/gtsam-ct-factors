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
      OptionalMatrixVecType H = nullptr) const {
    // if t_tau is equal to t_k or t_kp1, return the corresponding pose and
    // velocity
    if (equal(t_tau, t_k)) {
      return Tvarpi_k;
    } else if (equal(t_tau, t_kp1)) {
      return Tvarpi_kp1;
    } else {
      assert(t_tau >= t_k && t_tau <= t_kp1 &&
             "Query time must be between t_k and t_kp1");
    }

    auto [T_k, varpi_k] = Tvarpi_k;
    auto [T_kp1, varpi_kp1] = Tvarpi_kp1;

    Matrix2N Lambda, Psi;  // ensure Lambda and Psi are 2*dim x 2*dim matrices,
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
    // Eq. (11.45) in the book
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
    auto varpi_tau = right_jac_tau * (Lambda_2 * gamma_k + Psi_2 * gamma_kp1);

    // Compute Jacobians
    if (H) {
      // Derivative of velocity error wrt xi
      // Zero for vector spaces, use an approximation for Lie groups
      MatrixN dxidot_dxi;
      if constexpr (std::is_same_v<typename traits<PoseType>::structure_category, vector_space_tag>) {
        dxidot_dxi.setZero();
      } else {
        // For Lie groups
        dxidot_dxi = -PoseType::adjointMap(varpi_kp1) / 2.0;
      }
      // dxidot_dxi = 0*dxidot_dxi;
      
      // dgammakp1
      Eigen::Matrix<double, 2 * dim, dim> dgammakp1_dTk;
      dgammakp1_dTk << dxi_dTk, dxidot_dxi * dxi_dTk;  
      Eigen::Matrix<double, 2 * dim, dim> dgammakp1_dTkp1;
      dgammakp1_dTkp1 << dxi_dTkp1, dxidot_dxi * dxi_dTkp1;  
      Eigen::Matrix<double, 2 * dim, dim> dgammakp1_dvarpikp1;
      dgammakp1_dvarpikp1 << Eigen::Matrix<double, dim, dim>::Zero(),
          right_jac_inv;

      // dTtau_dTk
      (*H)[0] = dTtau_dTk + dTtau_dxitau * Psi_1 * dgammakp1_dTk;
      // dTtau_dvarpik
      (*H)[1] = dTtau_dxitau * Lambda_1.template block<dim,dim>(0,dim);
      // dTtau_dTkp1
      (*H)[2] = dTtau_dxitau * Psi_1 * dgammakp1_dTkp1;
      // dTtau_dvarpikp1
      (*H)[3] = dTtau_dxitau * Psi_1 * dgammakp1_dvarpikp1;
      // dvarpitau_dTk
      (*H)[4] = right_jac_tau * Psi_2 * dgammakp1_dTk;
      // dvarpitau_dvarpik
      (*H)[5] = right_jac_tau * Lambda_2 * dgammakp1_dTk;
      // dvarpitau_dTkp1
      (*H)[6] = right_jac_tau * Psi_2 * dgammakp1_dTkp1;
      // dvarpitau_dvarpikp1
      (*H)[7] = right_jac_tau * Psi_2 * dgammakp1_dvarpikp1;
    }

    return std::make_pair(T_tau, varpi_tau);
  }

  Values interpolatePosesAndVelocities(
      const NonlinearFactorGraph& mainSolveGraph,
      const Values& mainSolveSolution, const TimestampKeyMap& mainSolveKeyMap,
      const TimestampKeyMap& interpolateKeyMap,
      std::shared_ptr<CovarianceMap> covarianceMap) {
    // Map from intervals [t1, t2) to query times inside that interval (bucket)
    std::map<std::pair<double, double>, std::vector<double>> queryBuckets;

    for (const auto& [t_tau, keys] : interpolateKeyMap) {
      if (t_tau < mainSolveKeyMap.begin()->first ||
          t_tau > mainSolveKeyMap.rbegin()->first) {
        std::cerr
            << "Query time " << t_tau
            << " is out of bounds of the provided timestamps ranging from "
            << mainSolveKeyMap.begin()->first << " to "
            << mainSolveKeyMap.rbegin()->first << std::endl;
        throw std::out_of_range(
            "Query time is out of bounds of the provided timestamps.");
      }
      auto it2 = mainSolveKeyMap.upper_bound(t_tau);
      auto it1 = std::prev(it2);
      queryBuckets[std::make_pair(it1->first, it2->first)].push_back(t_tau);
    }

    Values interpolatedSolution;

    std::unique_ptr<Marginals>
        marginals;  // Only construct a Marginals if requested
    if (covarianceMap) {
      marginals =
          std::make_unique<Marginals>(mainSolveGraph, mainSolveSolution);
    }
    for (const auto& [interval, times] : queryBuckets) {
      double t_k = interval.first;
      double t_kp1 = interval.second;

      // Get the poses and velocities at t_k and t_kp1
      auto pvk = std::make_pair(
          mainSolveSolution.at<PoseType>(mainSolveKeyMap.at(t_k).first),
          mainSolveSolution.at<VelocityType>(mainSolveKeyMap.at(t_k).second));
      auto pvkp1 = std::make_pair(
          mainSolveSolution.at<PoseType>(mainSolveKeyMap.at(t_kp1).first),
          mainSolveSolution.at<VelocityType>(mainSolveKeyMap.at(t_kp1).second));

      // Interpolate for all query times within this query interval (bucket)
      for (double t_tau : times) {
        auto pvtau = interpolatePoseAndVelocity(pvk, t_k, pvkp1, t_kp1, t_tau);
        auto [T_tau, varpi_tau] = pvtau;
        interpolatedSolution.insert(interpolateKeyMap.at(t_tau).first, T_tau);
        interpolatedSolution.insert(interpolateKeyMap.at(t_tau).second,
                                    varpi_tau);
      }

      // Compute covariances of the interpolated poses and velocities
      if (covarianceMap) {
        // following (5.22) in paper
        KeyVector variables = {
            // {p1, v1, p2, v2}
            mainSolveKeyMap.at(t_k).first, mainSolveKeyMap.at(t_k).second,
            mainSolveKeyMap.at(t_kp1).first, mainSolveKeyMap.at(t_kp1).second};
        JointMarginal mainSolveMarginal =
            marginals->jointMarginalCovariance(variables);
        // avoid using JointMarginal.fullMatrix() as it returns covariance
        // in alphabetical order of the keys...
        Eigen::Matrix<double, 4 * dim, 4 * dim> mainSolveMarginalMatrix =
            constructMatrixFromJointMarginal(mainSolveMarginal, variables, dim);

        for (double t_tau : times) {
          auto pvtau = std::make_pair(interpolatedSolution.at<PoseType>(
                                          interpolateKeyMap.at(t_tau).first),
                                      interpolatedSolution.at<VelocityType>(
                                          interpolateKeyMap.at(t_tau).second));
          Matrix2N Sigma =
              computeConditionalCov(pvk, pvkp1, pvtau, t_k, t_kp1, t_tau);
          auto [Lambda, Psi] = getLamdaPsi(
              t_k, t_kp1, t_tau);  // todo: don't recompute this - already
                                   // computed in interpolation step
          Eigen::Matrix<double, 2 * dim, 4 * dim> LambdaPsi;
          LambdaPsi << Lambda, Psi;
          Matrix2N cov = Sigma + LambdaPsi * mainSolveMarginalMatrix *
                                     LambdaPsi.transpose();

          // upper left covariance block corresponds to pose, lower right block
          // corresponds to velocity
          covarianceMap->insert(
              {interpolateKeyMap.at(t_tau).first, cov.topLeftCorner(dim, dim)});
          covarianceMap->insert({interpolateKeyMap.at(t_tau).second,
                                 cov.bottomRightCorner(dim, dim)});
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

    // Eq. (11.41) in the book - can also use (5.23) in the paper
    auto Lambda =
        Phi_1tau - Q_1tau * Phi_tau2.transpose() * Q_12.inverse() * Phi_12;
    auto Psi = Q_1tau * Phi_tau2.transpose() * Q_12.inverse();

    return std::make_pair(Lambda, Psi);
  }

  Matrix2N computeConditionalCov(const std::pair<PoseType, VelocityType>& pvk,
                                 const std::pair<PoseType, VelocityType>& pvkp1,
                                 const std::pair<PoseType, VelocityType>& pvtau,
                                 double t_k, double t_kp1, double t_tau) const {
    Matrix2N Q_tau_prev = covarianceFunction_(t_tau - t_k, Q_psd_);
    Matrix2N Q_tau_next = covarianceFunction_(t_kp1 - t_tau, Q_psd_);
    Matrix2N E_tau = computeJacobianPrev_(pvk, pvtau, t_tau - t_k);
    Matrix2N F_tau = computeJacobianNext_(pvtau, pvkp1, t_kp1 - t_tau);
    Matrix2N Sigma_inv = E_tau.transpose() * Q_tau_prev.inverse() * E_tau +
                         F_tau.transpose() * Q_tau_next.inverse() * F_tau;
    Matrix2N Sigma = Sigma_inv.inverse();
    return Sigma;
  }

  static Matrix constructMatrixFromJointMarginal(
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
