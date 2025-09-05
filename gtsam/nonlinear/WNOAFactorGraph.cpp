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
namespace {

#ifdef GTSAM_USE_TBB
class _LinearizeOneFactor {
  const NonlinearFactorGraph& nonlinearGraph_;
  const Values& linearizationPoint_;
  GaussianFactorGraph& result_;
public:
  // Create functor with constant parameters
  _LinearizeOneFactor(const NonlinearFactorGraph& graph,
      const Values& linearizationPoint, GaussianFactorGraph& result) :
      nonlinearGraph_(graph), linearizationPoint_(linearizationPoint), result_(result) {
  }
  // Operator that linearizes a given range of the factors
  void operator()(const tbb::blocked_range<size_t>& blocked_range) const {
    for (size_t i = blocked_range.begin(); i != blocked_range.end(); ++i) {
      if (nonlinearGraph_[i] && nonlinearGraph_[i]->sendable())
        result_[i] = nonlinearGraph_[i]->linearize(linearizationPoint_);
      else
        result_[i] = GaussianFactor::shared_ptr();
    }
  }
};
#endif

}

/* ************************************************************************* */
template <typename PoseType>
GaussianFactorGraph::shared_ptr WNOAFactorGraph<PoseType>::linearize(const Values& linearizationPoint) const
{
  gttic(WNOAFactorGraph_linearize);

  // create an empty linear FG
  GaussianFactorGraph::shared_ptr linearFG = std::make_shared<GaussianFactorGraph>();

#ifdef GTSAM_USE_TBB

  linearFG->resize(size());
  TbbOpenMPMixedScope threadLimiter; // Limits OpenMP threads since we're mixing TBB and OpenMP

  // First linearize all sendable factors
  tbb::parallel_for(tbb::blocked_range<size_t>(0, size()),
    _LinearizeOneFactor(*this, linearizationPoint, *linearFG));

  // Linearize all non-sendable factors
  for(size_t i = 0; i < size(); i++) {
    auto& factor = (*this)[i];
    if(factor && !(factor->sendable())) {
      (*linearFG)[i] = factor->linearize(linearizationPoint);
    }
  }

#else

  linearFG->reserve(size());

  // linearize all factors
  for (const sharedFactor& factor : factors_) {
    if (factor) {
      linearFG->push_back(factor->linearize(linearizationPoint));
    } else
      linearFG->push_back(GaussianFactor::shared_ptr());
  }

#endif

  return linearFG;
}


} // namespace gtsam
