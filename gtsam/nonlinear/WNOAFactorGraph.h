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
#include <gtsam/nonlinear/StateData.h>
#include <gtsam/linear/GaussianFactorGraph.h>

using namespace std;

namespace gtsam {

/**
 * Factor graph that supports adding ExpressionFactors directly
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
  unordered_map<StateData, pair<StateData, StateData>> interp_to_borders_map_;
  bool fixed_noise_model_ = false;

  Values getInterpolatedValues(
      const Values& values,
      unordered_map<Key, unordered_map<Key, Matrix>>* InterpJacobians,
      unordered_map<StateData, Matrix2N>* InterpCondCovs = nullptr) const;

public:
    /// Linearize a nonlinear factor graph
    // Override to ensure this class's implementation is called, not the base class's
    std::shared_ptr<GaussianFactorGraph> linearize(const Values& linearizationPoint) const;
    // Constructor that initializes the interpolator and interp_to_borders_map_
    WNOAFactorGraph(unordered_map<StateData, pair<StateData, StateData>> interp_map, const Eigen::Vector<double, dim> Q_psd, bool fixed_noise_model = false)
        : interpolator_(Q_psd),
          interp_to_borders_map_(std::move(interp_map)),
          fixed_noise_model_(fixed_noise_model) {}

};

}
