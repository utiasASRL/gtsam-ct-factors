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
#include <unordered_map>
#include <gtsam/nonlinear/StateData.h>

namespace gtsam {

/**
 * Factor graph that supports adding ExpressionFactors directly
 */
class WNOAFactorGraph: public WNOAFactorGraph {

private:
  // Caches for interpolated values, Jacobians, and conditional covariances
  Values* InterpValuesCache = nullptr;
  std::unordered_map<Key, std::unordered_map<Key, Matrix>>* InterpJacobiansCache = nullptr;
  std::unordered_map<StateData, Matrix>* InterpCondCovsCache = nullptr;

public:
    /// Linearize a nonlinear factor graph
    std::shared_ptr<GaussianFactorGraph> linearize(const Values& linearizationPoint) const;

};

}
