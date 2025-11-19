/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 *  @file  WNOAFactorGraph.h
 *  @brief Factor graph that handles computation of interpolated states for WNOA
 *  @author Sven Lilge
 */

#pragma once

#include <gtsam/nonlinear/ExpressionFactorGraph.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Interpolator.h>
#include <unordered_map>
#include <unordered_set>
#include <gtsam/nonlinear/StateData.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <array>

using namespace std;

namespace gtsam {

/**
 * Factor graph that handles computation of interpolated states for WNOA
 */
template <typename PoseType>
class WNOAFactorGraph: public ExpressionFactorGraph {

private:

  using This = WNOAFactorGraph<PoseType>;
  using Base = ExpressionFactorGraph;
  using VelocityType = typename gtsam::traits<PoseType>::TangentVector;
  static constexpr int dim = traits<PoseType>::dimension;

  // Convenient matrices
  using Matrix2N = Eigen::Matrix<double, 2 * dim, 2 * dim>;
  using MatrixN = Eigen::Matrix<double, dim, dim>;


  // Interpolator class
  const Interpolator<PoseType> interpolator_;

  // map interpolated state to border states
  unordered_map<StateData, pair<StateData, StateData>> interp_to_borders_map_;
  std::vector<std::pair<StateData, std::pair<StateData, StateData>>> interp_to_borders_vec_;

  bool fixed_noise_model_ = false;

  unordered_set<Key> border_pose_keys_;
  unordered_set<Key> border_vel_keys_;


  // Efficient storage for indices of WNOAInterpFactors
  unordered_set<size_t> wnoa_interp_factor_indices_;

  Values getInterpolatedValues(
      const Values& values,
    unordered_map<Key, std::array<Matrix, 4>>* InterpJacobians,
    unordered_map<StateData, Matrix2N>* InterpCondCovs = nullptr) const;

public:
    /// Linearize a nonlinear factor graph exploiting precomputation of interpolation data
    std::shared_ptr<GaussianFactorGraph> linearize(const Values& linearizationPoint) const;

    /** unnormalized error, \f$ \sum_i 0.5 (h_i(X_i)-z)^2 / \sigma^2 \f$ in the most common case - exploiting precomputation of interpolation data*/
    double error(const Values& values) const;

    // TODO: These vectors need to be defined to indicate which factors are WNOAInterpFactors
    // Switch back to casting factors to detect this, in order to avoid needing to define these vectors for the user?
    void setWNOAInterpFactorIndices(const unordered_set<size_t>& indices) {
      wnoa_interp_factor_indices_ = indices;
    }

    bool isWNOAInterpFactorIndex(size_t index) const {
      return wnoa_interp_factor_indices_.count(index) > 0;
    }


    // Constructor that initializes the interpolator and interp_to_borders_map_
    WNOAFactorGraph(unordered_map<StateData, pair<StateData, StateData>> interp_map, const Eigen::Vector<double, dim> Q_psd, bool fixed_noise_model = false)
        : interpolator_(Q_psd),
          interp_to_borders_map_(std::move(interp_map)),
          fixed_noise_model_(fixed_noise_model) {
            // Collect unique keys for boundary states to avoid repeated Values::at lookups.
            border_pose_keys_.reserve(interp_to_borders_map_.size()*2);
            border_vel_keys_.reserve(interp_to_borders_map_.size()*2);
            for (const auto& kv : interp_to_borders_map_) {
              const auto& border_states = kv.second;
              const auto& left  = border_states.first;
              const auto& right = border_states.second;
              border_pose_keys_.insert(left.pose);  border_pose_keys_.insert(right.pose);
              border_vel_keys_.insert(left.vel);    border_vel_keys_.insert(right.vel);
            }

            // Convert map to vector for optimal parallel access patterns
            interp_to_borders_vec_ = std::vector<std::pair<StateData, std::pair<StateData, StateData>>>(
                interp_to_borders_map_.begin(), interp_to_borders_map_.end());

          }

};

}
