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

#include <gtsam/nonlinear/WNOAFactorGraph.h>
#include <gtsam/nonlinear/WNOAInterpFactor.h>

#ifdef GTSAM_USE_TBB
#  include <tbb/parallel_for.h>
#endif

using namespace std;

namespace gtsam {


/* ************************************************************************* */
GaussianFactorGraph::shared_ptr WNOAFactorGraph::linearize(const Values& linearizationPoint) const
{

  // Compute interpolated values, Jacobians, and conditional covariances and cache values

  // We need to run through all factors in the graph and check which ones are of type WNOAInterpFactor
  for (const auto& factor : factors_) {

    unordered_map<Key, StateData> key_to_interp_factor;
    unordered_map<StateData, pair<StateData, StateData>> interp_to_borders_factor;
    // Downcast the NonlinearFactor to a WNOAInterpFactor
    if(auto wnoa_interp_factor = dynamic_pointer_cast<WNOAInterpFactor<Pose3>>(factor))
    {
      key_to_interp_factor = wnoa_interp_factor->getInterpolatedKeys();
      interp_to_borders_factor = wnoa_interp_factor->getInterpToBorders();
    }
    else if(auto wnoa_interp_factor = dynamic_pointer_cast<WNOAInterpFactor<Pose2>>(factor))
    {
      key_to_interp_factor = wnoa_interp_factor->getInterpolatedKeys();
      interp_to_borders_factor = wnoa_interp_factor->getInterpToBorders();
    }
    else if(auto wnoa_interp_factor = dynamic_pointer_cast<WNOAInterpFactor<Point3>>(factor))
    {
      key_to_interp_factor = wnoa_interp_factor->getInterpolatedKeys();
      interp_to_borders_factor = wnoa_interp_factor->getInterpToBorders();
    }
    else if(auto wnoa_interp_factor = dynamic_pointer_cast<WNOAInterpFactor<Point2>>(factor))
    {
      key_to_interp_factor = wnoa_interp_factor->getInterpolatedKeys();
      interp_to_borders_factor = wnoa_interp_factor->getInterpToBorders();
    }
    else if(auto wnoa_interp_factor = dynamic_pointer_cast<WNOAInterpFactor<Point1>>(factor))
    {
      key_to_interp_factor = wnoa_interp_factor->getInterpolatedKeys();
      interp_to_borders_factor = wnoa_interp_factor->getInterpToBorders();
    }
    else
    {
      // Factor is not a WNOAInterpFactor, skip
      continue;
    }

    // Add to graph-level map if not already present
    for (const auto& [key, state] : key_to_interp_factor) {
      if (key_to_interp_graph_.count(key) == 0) {
        key_to_interp_graph_[key] = state;
      }
    }
    for (const auto& [state, borders] : interp_to_borders_factor) {
      if (interp_to_borders_graph_.count(state) == 0) {
        interp_to_borders_graph_[state] = borders;
      }
    }


  // Run base class routine
  return NonlinearFactorGraph::linearize(linearizationPoint);

  }
}
} // namespace gtsam
