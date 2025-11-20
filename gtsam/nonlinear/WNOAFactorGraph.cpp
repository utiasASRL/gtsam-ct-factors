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
#include <gtsam/config.h> // for GTSAM_USE_TBB
#include <Eigen/Core> // for Eigen::setNbThreads

#ifdef GTSAM_USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/partitioner.h>
#endif

#include <mutex>
#include <unordered_set>
#include <algorithm>
#include <array>
#include <thread>
#include <memory>


using namespace std;

namespace gtsam {

  /* ************************************************************************* */
namespace {


#ifdef GTSAM_USE_TBB
template <typename PoseType>
class _LinearizeOneFactor {
  const WNOAFactorGraph<PoseType>& nonlinearGraph_;
  const Values& linearizationPoint_;
  GaussianFactorGraph& result_;
  typename WNOAInterpFactor<PoseType>::PassedInterpData& passedInterpData_;

public:
  // Create functor with constant parameters
  _LinearizeOneFactor(const WNOAFactorGraph<PoseType>& graph,
      const Values& linearizationPoint, GaussianFactorGraph& result,
      typename WNOAInterpFactor<PoseType>::PassedInterpData& passedInterpData) :
      nonlinearGraph_(graph), linearizationPoint_(linearizationPoint), result_(result),
      passedInterpData_(passedInterpData) {
  }
  // Operator that linearizes a given range of the factors
  void operator()(const tbb::blocked_range<size_t>& blocked_range) const {
    for (size_t i = blocked_range.begin(); i != blocked_range.end(); ++i) {
      if (nonlinearGraph_[i] && nonlinearGraph_[i]->sendable())
      {
        // Attempt dynamic cast to WNOAInterpFactor
        auto wnoa_factor = dynamic_pointer_cast<WNOAInterpFactor<PoseType>>(nonlinearGraph_[i]);
        if (wnoa_factor) {
          result_[i] = wnoa_factor->linearize(linearizationPoint_, &passedInterpData_);
        } else {
          result_[i] = nonlinearGraph_[i]->linearize(linearizationPoint_);
        }
      }
      else
      {
        result_[i] = GaussianFactor::shared_ptr();
      }
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

  // gttic(WNOAFactorGraph_linearize_InterpValues);


  // Compute values, Jacobians and conditional covariances for all interpolated states

  auto passedInterpData = std::make_shared<typename WNOAInterpFactor<PoseType>::PassedInterpData>();

  if(fixed_noise_model_) {
    passedInterpData->values = getInterpolatedValues(linearizationPoint, &passedInterpData->jacobians, nullptr);
  } else {
    passedInterpData->values = getInterpolatedValues(linearizationPoint, &passedInterpData->jacobians, &passedInterpData->condCovs);
  }


  // gttoc(WNOAFactorGraph_linearize_InterpValues);

  
  // gttic(WNOAFactorGraph_linearize_factors);

  // create an empty linear FG
  GaussianFactorGraph::shared_ptr linearFG = std::make_shared<GaussianFactorGraph>();

#ifdef GTSAM_USE_TBB

  linearFG->resize(size());
  TbbOpenMPMixedScope threadLimiter; // Limits OpenMP threads since we're mixing TBB and OpenMP

  // First linearize all sendable factors (hint affinity for better cache locality)
  {
    tbb::affinity_partitioner ap;
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, size()),
        _LinearizeOneFactor<PoseType>(*this, linearizationPoint, *linearFG, *passedInterpData),
        ap);
  }

  // Linearize all non-sendable factors
  for(size_t i = 0; i < size(); i++) {
    auto& factor = (*this)[i];
    if(factor && !(factor->sendable())) {
      // Attempt dynamic cast to WNOAInterpFactor
      auto wnoa_factor = dynamic_pointer_cast<WNOAInterpFactor<PoseType>>(factor);
      if (wnoa_factor) {
        (*linearFG)[i] = wnoa_factor->linearize(linearizationPoint, passedInterpData.get());
      } else {
        (*linearFG)[i] = factor->linearize(linearizationPoint);
      }
    }
  }

#else

  linearFG->reserve(size());

  // linearize all factors
  for (const sharedFactor& factor : factors_) {
    // Attempt dynamic cast to WNOAInterpFactor
    if (factor) {
      auto wnoa_factor = dynamic_pointer_cast<WNOAInterpFactor<PoseType>>(factor);
      if (wnoa_factor) {
        linearFG->push_back(wnoa_factor->linearize(linearizationPoint, passedInterpData.get()));
      } else {
      linearFG->push_back(factor->linearize(linearizationPoint));
      }
    } else
      linearFG->push_back(GaussianFactor::shared_ptr());
  }

#endif

  //gttoc(WNOAFactorGraph_linearize_factors);

  return linearFG;
}


template <typename PoseType>
double WNOAFactorGraph<PoseType>::error(const Values& values) const {
  gttic(WNOAFactorGraph_error);


  double total_error = 0.;

  // gttic(WNOAFactorGraph_error_InterpValues);
  // Compute values, Jacobians and conditional covariances for all interpolated states

  auto passedInterpData = std::make_shared<typename WNOAInterpFactor<PoseType>::PassedInterpData>();
  if(fixed_noise_model_) {
    passedInterpData->values = getInterpolatedValues(values, nullptr, nullptr);
  } else {
    passedInterpData->values = getInterpolatedValues(values, nullptr, &passedInterpData->condCovs);
  }
  // gttoc(WNOAFactorGraph_error_InterpValues);

  // gttic(WNOAFactorGraph_error_factors);

  // iterate over all the factors_ to accumulate the log probabilities
  for(const sharedFactor& factor: factors_) {

    if (factor) {
      auto wnoa_factor = dynamic_pointer_cast<WNOAInterpFactor<PoseType>>(factor);
      if (wnoa_factor) {
        total_error += wnoa_factor->error(values, passedInterpData.get());
      } else {
        total_error += factor->error(values);
      }
    }
  }

  // gttoc(WNOAFactorGraph_error_factors);
  return total_error;
}

/* @brief Interpolate all interpolated states based on estimated states.
  * Put their values in a Values structure and compute their Jacobians.*/
template <typename PoseType>
Values WNOAFactorGraph<PoseType>::getInterpolatedValues(
  const Values& values,
  unordered_map<Key, std::array<Matrix, 4>>* InterpJacobians,
  unordered_map<StateData, Matrix2N>* InterpCondCovs) const {
  // Refactored: iterate over border pairs, compute shared left/right data once, then all interp states.
#ifdef GTSAM_USE_TBB
  TbbOpenMPMixedScope threadLimiter;
  // Cache boundary values
  unordered_map<Key, PoseType>    border_pose_cache; border_pose_cache.reserve(border_pose_keys_.size());
  unordered_map<Key, VelocityType> border_vel_cache; border_vel_cache.reserve(border_vel_keys_.size());
  for (Key k : border_pose_keys_) border_pose_cache.emplace(k, values.at<PoseType>(k));
  for (Key k : border_vel_keys_)  border_vel_cache.emplace(k, values.at<VelocityType>(k));

  // Use precomputed border batches stored in the graph (built in constructor)
  auto &batches = this->border_batches_;

  struct TLS { Values local_values; std::vector<std::pair<Key,std::array<Matrix,4>>> jacs; std::vector<std::pair<StateData,Matrix2N>> condcovs; std::vector<Matrix> H; };
  tbb::enumerable_thread_specific<TLS> tls;

  unsigned nt = std::thread::hardware_concurrency(); unsigned phys = nt? std::max<unsigned>(1u, nt/2):1u;
  size_t grain = std::max<size_t>(1, batches.size() / (8*phys));
  {
    tbb::affinity_partitioner ap;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, batches.size(), grain), [&](const tbb::blocked_range<size_t>& r){
      auto &local = tls.local(); if (local.H.empty()) local.H.resize(8);
      size_t expectedInterp = 0; for (size_t bi=r.begin(); bi!=r.end(); ++bi) expectedInterp += batches[bi].second.size();
      if (InterpJacobians) local.jacs.reserve(local.jacs.size()+expectedInterp*2);
      if (InterpCondCovs) local.condcovs.reserve(local.condcovs.size()+expectedInterp);
      for (size_t bi=r.begin(); bi!=r.end(); ++bi) {
        const auto &bp = batches[bi].first; const StateData &left = bp.first; const StateData &right = bp.second;
        const auto state_left = TimestampedPoseVelocity<PoseType>(border_pose_cache[left.pose], border_vel_cache[left.vel], left.time);
        const auto state_right= TimestampedPoseVelocity<PoseType>(border_pose_cache[right.pose], border_vel_cache[right.vel], right.time);

        // Precompute local state vars and local-global Jacobians for this border pair
        std::shared_ptr<LocalGlobalStateJacs> localGlobalStateJacsPreComp = std::make_shared<LocalGlobalStateJacs>();
        std::shared_ptr<LocalStateVecs> localStateVecsPreComp = std::make_shared<LocalStateVecs>();

        if (InterpJacobians) {
          *localStateVecsPreComp = interpolator_.computeLocalStateVecs(state_left, state_right, localGlobalStateJacsPreComp.get());
        } else {
          *localStateVecsPreComp = interpolator_.computeLocalStateVecs(state_left, state_right, nullptr);
        }
        

        for (size_t interpIdx : batches[bi].second) {
          const StateData &interp_state = interp_to_borders_vec_[interpIdx].first;
          PoseVelocity<PoseType> result;
          if (InterpJacobians) {
            result = interpolator_.interpolatePoseAndVelocity(state_left, state_right, interp_state.time, &local.H, nullptr, nullptr, interp_to_LambdaPsi_vec_[interpIdx].second, localStateVecsPreComp, localGlobalStateJacsPreComp);
            std::array<Matrix,4> Jpose = {local.H[0],local.H[1],local.H[2],local.H[3]};
            std::array<Matrix,4> Jvel  = {local.H[4],local.H[5],local.H[6],local.H[7]};
            local.jacs.emplace_back(interp_state.pose, std::move(Jpose));
            local.jacs.emplace_back(interp_state.vel,  std::move(Jvel));
          } else {
            result = interpolator_.interpolatePoseAndVelocity(state_left, state_right, interp_state.time, nullptr, nullptr, nullptr, interp_to_LambdaPsi_vec_[interpIdx].second, localStateVecsPreComp, nullptr);
          }
          local.local_values.insert(interp_state.pose, std::move(result.pose));
          local.local_values.insert(interp_state.vel, std::move(result.vel));
          if (InterpCondCovs) {
            auto state_tau = TimestampedPoseVelocity<PoseType>(result, interp_state.time);
            Matrix2N Sigma_tau = interpolator_.computeConditionalCov(state_left, state_right, state_tau);
            local.condcovs.emplace_back(interp_state, std::move(Sigma_tau));
          }
        }
      }
    }, ap);
  }
  Values values_interp;
  if (InterpJacobians) InterpJacobians->reserve(interp_to_borders_vec_.size()*2);
  if (InterpCondCovs) InterpCondCovs->reserve(interp_to_borders_vec_.size());
  for (auto &local : tls) {
    // Bulk-insert per-thread Values
    values_interp.insert(local.local_values);
    if (InterpJacobians) for (auto &kv : local.jacs) InterpJacobians->insert_or_assign(kv.first, std::move(kv.second));
    if (InterpCondCovs) for (auto &sc : local.condcovs) (*InterpCondCovs)[sc.first] = std::move(sc.second);
  }
  return values_interp;
#else
  // Sequential version
  unordered_map<Key, PoseType>    border_pose_cache; border_pose_cache.reserve(border_pose_keys_.size());
  unordered_map<Key, VelocityType> border_vel_cache; border_vel_cache.reserve(border_vel_keys_.size());
  for (Key k : border_pose_keys_) border_pose_cache.emplace(k, values.at<PoseType>(k));
  for (Key k : border_vel_keys_)  border_vel_cache.emplace(k, values.at<VelocityType>(k));
  std::vector<Matrix> H(8);
  Values values_interp;
  if (InterpJacobians) InterpJacobians->reserve(interp_to_borders_vec_.size()*2);
  if (InterpCondCovs) InterpCondCovs->reserve(interp_to_borders_vec_.size());
  for (const auto &kv : border_batches_) {
    const StateData &left = kv.first.first; const StateData &right = kv.first.second;
    const auto state_left = TimestampedPoseVelocity<PoseType>(border_pose_cache[left.pose], border_vel_cache[left.vel], left.time);
    const auto state_right= TimestampedPoseVelocity<PoseType>(border_pose_cache[right.pose], border_vel_cache[right.vel], right.time);
    
    // Precompute local state vars and local-global Jacobians for this border pair
    std::shared_ptr<LocalGlobalStateJacs> localGlobalStateJacsPreComp = std::make_shared<LocalGlobalStateJacs>();
    std::shared_ptr<LocalStateVecs> localStateVecsPreComp = std::make_shared<LocalStateVecs>();

    if (InterpJacobians) {
      *localStateVecsPreComp = interpolator_.computeLocalStateVecs(state_left, state_right, localGlobalStateJacsPreComp.get());
    } else {
      *localStateVecsPreComp = interpolator_.computeLocalStateVecs(state_left, state_right, nullptr);
    }
    

    
    for (size_t interpIdx : kv.second) {
      const StateData &interp_state = interp_to_borders_vec_[interpIdx].first;
      PoseVelocity<PoseType> result;
      if (InterpJacobians) {
        result = interpolator_.interpolatePoseAndVelocity(state_left, state_right, interp_state.time, &H, nullptr, nullptr, interp_to_LambdaPsi_vec_[interpIdx].second, localStateVecsPreComp, localGlobalStateJacsPreComp);
      } else {
        result = interpolator_.interpolatePoseAndVelocity(state_left, state_right, interp_state.time, nullptr, nullptr, nullptr, interp_to_LambdaPsi_vec_[interpIdx].second, localStateVecsPreComp, nullptr);
      }
      values_interp.insert(interp_state.pose, result.pose);
      values_interp.insert(interp_state.vel,  result.vel);
      if (InterpJacobians) {
        (*InterpJacobians)[interp_state.pose] = std::array<Matrix,4>{H[0],H[1],H[2],H[3]};
        (*InterpJacobians)[interp_state.vel]  = std::array<Matrix,4>{H[4],H[5],H[6],H[7]};
      }
      if (InterpCondCovs) {
        auto state_tau = TimestampedPoseVelocity<PoseType>(result, interp_state.time);
        Matrix2N Sigma_tau = interpolator_.computeConditionalCov(state_left, state_right, state_tau);
        (*InterpCondCovs)[interp_state] = std::move(Sigma_tau);
      }
    }
  }
  return values_interp;
#endif
}

// Explicit template instantiation
template class WNOAFactorGraph<Point1>;
template class WNOAFactorGraph<Point2>;
template class WNOAFactorGraph<Point3>;
template class WNOAFactorGraph<Pose2>;
template class WNOAFactorGraph<Pose3>;

} // namespace gtsam
