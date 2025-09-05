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

  unordered_map<StateData, pair<StateData, StateData>> interp_to_borders_map_;
  unordered_map<size_t, set<StateData>> factor_to_interp_map_;

  void initializeMaps(unordered_map<StateData, pair<StateData, StateData>> interp_map,
                      unordered_map<size_t, set<StateData>> factor_map) {
    interp_to_borders_map_ = std::move(interp_map);
    factor_to_interp_map_ = std::move(factor_map);
  }

public:
    /// Linearize a nonlinear factor graph
    std::shared_ptr<GaussianFactorGraph> linearize(const Values& linearizationPoint) const;

};

}
