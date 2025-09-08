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
template <typename PoseType>
class _LinearizeOneFactor {
  const NonlinearFactorGraph& nonlinearGraph_;
  const Values& linearizationPoint_;
  GaussianFactorGraph& result_;
  typename WNOAInterpFactor<PoseType>::PassedInterpData& passedInterpData_;

public:
  // Create functor with constant parameters
  _LinearizeOneFactor(const NonlinearFactorGraph& graph,
      const Values& linearizationPoint, GaussianFactorGraph& result,
      typename WNOAInterpFactor<PoseType>::PassedInterpData& passedInterpData) :
      nonlinearGraph_(graph), linearizationPoint_(linearizationPoint), result_(result),
      passedInterpData_(passedInterpData) {
  }
  // Operator that linearizes a given range of the factors
  void operator()(const tbb::blocked_range<size_t>& blocked_range) const {
    for (size_t i = blocked_range.begin(); i != blocked_range.end(); ++i) {
      if (nonlinearGraph_[i] && nonlinearGraph_[i]->sendable())
        if (auto wnoa_factor = dynamic_pointer_cast<WNOAInterpFactor<PoseType>>(nonlinearGraph_[i])) {
          result_[i] = wnoa_factor->linearizePassedInterpData(linearizationPoint_, &passedInterpData_);
        } else {
          result_[i] = nonlinearGraph_[i]->linearize(linearizationPoint_);
        }
      else
        result_[i] = GaussianFactor::shared_ptr();
    }
  }
};
#endif

}

/* ************************************************************************* */
template <typename PoseType>
std::shared_ptr<GaussianFactorGraph>  WNOAFactorGraph<PoseType>::linearize(const Values& linearizationPoint) const
{
  gttic(WNOAFactorGraph_linearize);


  // Compute values, Jacobians and conditional covariances for all interpolated states

  Values InterpValues;
  unordered_map<Key, unordered_map<Key, Matrix>> InterpJacobians;
  unordered_map<StateData, Matrix2N> InterpCondCovs;
  if(fixed_noise_model_) {
    InterpValues = getInterpolatedValues(linearizationPoint, &InterpJacobians, nullptr);
  } else {
    InterpValues = getInterpolatedValues(linearizationPoint, &InterpJacobians, &InterpCondCovs);
  }

  
  auto passedInterpData = std::make_shared<typename WNOAInterpFactor<PoseType>::PassedInterpData>();
  passedInterpData->values = InterpValues;
  passedInterpData->jacobians = InterpJacobians;
  passedInterpData->condCovs = InterpCondCovs;


  // create an empty linear FG
  GaussianFactorGraph::shared_ptr linearFG = std::make_shared<GaussianFactorGraph>();

#ifdef GTSAM_USE_TBB

  linearFG->resize(size());
  TbbOpenMPMixedScope threadLimiter; // Limits OpenMP threads since we're mixing TBB and OpenMP

  // First linearize all sendable factors
  tbb::parallel_for(tbb::blocked_range<size_t>(0, size()),
    _LinearizeOneFactor<PoseType>(*this, linearizationPoint, *linearFG, *passedInterpData));

  // Linearize all non-sendable factors
  for(size_t i = 0; i < size(); i++) {
    auto& factor = (*this)[i];
    if(factor && !(factor->sendable())) {
      if (auto wnoa_factor = dynamic_pointer_cast<WNOAInterpFactor<PoseType>>(factor)) {
        (*linearFG)[i] = wnoa_factor->linearizePassedInterpData(linearizationPoint, passedInterpData.get());
      } else {
        (*linearFG)[i] = factor->linearize(linearizationPoint);
      }
    }
  }

#else

  linearFG->reserve(size());

  // linearize all factors
  for (const sharedFactor& factor : factors_) {
    if (factor) {
      if (auto wnoa_factor = dynamic_pointer_cast<WNOAInterpFactor<PoseType>>(factor)) {
        linearFG->push_back(wnoa_factor->linearizePassedInterpData(linearizationPoint, passedInterpData.get()));
      } else {
        linearFG->push_back(factor->linearize(linearizationPoint));
      }
    } else
      linearFG->push_back(GaussianFactor::shared_ptr());
  }

#endif

  return linearFG;
}


/* @brief Interpolate all interpolated states based on estimated states.
  * Put their values in a Values structure and compute their Jacobians.*/
template <typename PoseType>
Values WNOAFactorGraph<PoseType>::getInterpolatedValues(
    const Values& values,
    unordered_map<Key, unordered_map<Key, Matrix>>* InterpJacobians,
    unordered_map<StateData, Matrix2N>* InterpCondCovs) const {
  Values values_interp;  // interpolated values
  // loop through interpolated state map and compute values
  for (const auto& [interp_state, border_states] : interp_to_borders_map_) {
    // unpack border states
    auto& [left, right] = border_states;
    // retrieve estimated state values
    const auto state_left = TimestampedPoseVelocity<PoseType>(
        values.at<PoseType>(left.pose),
        values.at<VelocityType>(left.vel),
        left.time);

    const auto state_right = TimestampedPoseVelocity<PoseType>(
        values.at<PoseType>(right.pose),
        values.at<VelocityType>(right.vel),
        right.time);

    // Get interpolated state velocity pair
    PoseVelocity<PoseType> result;
    vector<Matrix> H(8);
    if (InterpJacobians) {
      result = interpolator_.interpolatePoseAndVelocity(
          state_left, state_right, interp_state.time,
          &H);
    } else {
      result = interpolator_.interpolatePoseAndVelocity(
          state_left, state_right, interp_state.time);
    }

    // insert into values structure
    values_interp.insert(interp_state.pose, result.pose);
    values_interp.insert(interp_state.vel, result.vel);

    // arrange jacobians in unordered map (for easy access later)
    if (InterpJacobians) {
      (*InterpJacobians)[interp_state.pose][left.pose] = H[0];
      (*InterpJacobians)[interp_state.pose][left.vel] = H[1];
      (*InterpJacobians)[interp_state.pose][right.pose] = H[2];
      (*InterpJacobians)[interp_state.pose][right.vel] = H[3];
      (*InterpJacobians)[interp_state.vel][left.pose] = H[4];
      (*InterpJacobians)[interp_state.vel][left.vel] = H[5];
      (*InterpJacobians)[interp_state.vel][right.pose] = H[6];
      (*InterpJacobians)[interp_state.vel][right.vel] = H[7];
    }

    // Conditional covariance of interpolated states for noise model update
    if (InterpCondCovs) {
      auto state_tau = TimestampedPoseVelocity<PoseType>(
          result, interp_state.time);
      Matrix2N Sigma_tau = interpolator_.computeConditionalCov(
          state_left, state_right, state_tau);
      (*InterpCondCovs)[interp_state] = Sigma_tau;  // assumed preallocated vector
    }
  }

  return values_interp;
}

// Explicit template instantiation
template class WNOAFactorGraph<Point1>;
template class WNOAFactorGraph<Point2>;
template class WNOAFactorGraph<Point3>;
template class WNOAFactorGraph<Pose2>;
template class WNOAFactorGraph<Pose3>;

} // namespace gtsam
