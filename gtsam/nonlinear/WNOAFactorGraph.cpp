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
#  include <tbb/parallel_for_each.h>
#  include <tbb/concurrent_unordered_map.h>
#  include <tbb/enumerable_thread_specific.h>
#endif

#include <mutex>
#include <unordered_set>


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
        // Check if i is in wnoa_interp_factor_indices_
        if(nonlinearGraph_.isWNOAInterpFactorIndex(i)) {
          // This is a WNOAInterpFactor, cast down statically to avoid dynamic cast
          auto wnoa_factor = static_pointer_cast<WNOAInterpFactor<PoseType>>(nonlinearGraph_[i]);
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

  gttic(WNOAFactorGraph_linearize_InterpValues);


  // Compute values, Jacobians and conditional covariances for all interpolated states

  auto passedInterpData = std::make_shared<typename WNOAInterpFactor<PoseType>::PassedInterpData>();

  if(fixed_noise_model_) {
    passedInterpData->values = getInterpolatedValues(linearizationPoint, &passedInterpData->jacobians, nullptr);
  } else {
    passedInterpData->values = getInterpolatedValues(linearizationPoint, &passedInterpData->jacobians, &passedInterpData->condCovs);
  }


  gttoc(WNOAFactorGraph_linearize_InterpValues);

  
  gttic(WNOAFactorGraph_linearize_factors);

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
      // Check if i is in wnoa_interp_factor_indices_
      size_t i = &factor - &factors_[0];
      if (isWNOAInterpFactorIndex(i)) {
        // This is a WNOAInterpFactor, cast down statically to avoid dynamic cast
        auto wnoa_factor = static_pointer_cast<WNOAInterpFactor<PoseType>>(factor);
        (*linearFG)[i] = wnoa_factor->linearize(linearizationPoint, passedInterpData.get());
      } else{
        (*linearFG)[i] = factor->linearize(linearizationPoint);
      }
    }
  }

#else

  linearFG->reserve(size());

  // linearize all factors
  for (const sharedFactor& factor : factors_) {
    // Get index of factor
    size_t i = &factor - &factors_[0];
    // Check if i is in wnoa_interp_factor_indices_
    if (isWNOAInterpFactorIndex(i)) {
      // This is a WNOAInterpFactor, cast down statically to avoid dynamic cast
      auto wnoa_factor = static_pointer_cast<WNOAInterpFactor<PoseType>>(factor);
      linearFG->push_back(wnoa_factor->linearize(linearizationPoint, passedInterpData.get()));
    } else if (factor) {
      linearFG->push_back(factor->linearize(linearizationPoint));
    } else
      linearFG->push_back(GaussianFactor::shared_ptr());
  }

#endif

  gttoc(WNOAFactorGraph_linearize_factors);

  return linearFG;
}


template <typename PoseType>
double WNOAFactorGraph<PoseType>::error(const Values& values) const {
  gttic(WNOAFactorGraph_error);


  double total_error = 0.;


  gttic(WNOAFactorGraph_error_InterpValues);
  // Compute values, Jacobians and conditional covariances for all interpolated states

  auto passedInterpData = std::make_shared<typename WNOAInterpFactor<PoseType>::PassedInterpData>();
  if(fixed_noise_model_) {
    passedInterpData->values = getInterpolatedValues(values, nullptr, nullptr);
  } else {
    passedInterpData->values = getInterpolatedValues(values, nullptr, &passedInterpData->condCovs);
  }
  gttoc(WNOAFactorGraph_error_InterpValues);

  gttic(WNOAFactorGraph_error_factors);

  // iterate over all the factors_ to accumulate the log probabilities
  for(const sharedFactor& factor: factors_) {

     size_t i = &factor - &factors_[0];
    // Check if i is in wnoa_interp_factor_indices_
    if (isWNOAInterpFactorIndex(i)) {
      // This is a WNOAInterpFactor, cast down statically to avoid dynamic cast
      auto wnoa_factor = static_pointer_cast<WNOAInterpFactor<PoseType>>(factor);
      total_error += wnoa_factor->error(values, passedInterpData.get());
    } else if (factor) {
      total_error += factor->error(values);
    }
  }

  gttoc(WNOAFactorGraph_error_factors);
  return total_error;
}

/* @brief Interpolate all interpolated states based on estimated states.
  * Put their values in a Values structure and compute their Jacobians.*/
template <typename PoseType>
Values WNOAFactorGraph<PoseType>::getInterpolatedValues(
    const Values& values,
    unordered_map<Key, unordered_map<Key, Matrix>>* InterpJacobians,
    unordered_map<StateData, Matrix2N>* InterpCondCovs) const {

#ifdef GTSAM_USE_TBB

  // --- Step 1: Cache boundary poses/velocities ---
    unordered_map<Key, PoseType>    border_pose_cache;
    unordered_map<Key, VelocityType> border_vel_cache;
    border_pose_cache.reserve(border_pose_keys_.size());
    border_vel_cache.reserve(border_vel_keys_.size());
    for (Key k : border_pose_keys_) border_pose_cache.emplace(k, values.at<PoseType>(k));
    for (Key k : border_vel_keys_)  border_vel_cache.emplace(k, values.at<VelocityType>(k));

    // --- Step 2: Define thread-local storage ---
    struct ThreadLocalData {
        std::vector<std::pair<Key, PoseType>> poses;
        std::vector<std::pair<Key, VelocityType>> vels;
        std::vector<std::tuple<Key, Key, Matrix>> jacobians;
        std::vector<std::pair<StateData, Matrix2N>> condcovs;
        vector<Matrix> H; // Reuse matrices per thread
    };

    tbb::enumerable_thread_specific<ThreadLocalData> thread_data;

    // Get number of physical cores (approximate if hyperthreaded)
    unsigned int num_logical = std::thread::hardware_concurrency();
    unsigned int num_physical_cores = num_logical > 0 ? std::max(1u, num_logical / 2) : 1;
    size_t grain = std::max(size_t(1), interp_to_borders_vec_.size() / (8 * num_physical_cores));
    // --- Step 3: Parallel interpolation ---
    tbb::parallel_for(tbb::blocked_range<size_t>(0, interp_to_borders_vec_.size(), grain),
        [&](const tbb::blocked_range<size_t>& range) {
            auto& local_data = thread_data.local();

            // Preallocate vectors based on chunk size
            size_t chunk_size = range.size();
            local_data.poses.reserve(chunk_size * 2);
            local_data.vels.reserve(chunk_size * 2);
            if (InterpJacobians) local_data.jacobians.reserve(chunk_size * 8);
            if (InterpCondCovs) local_data.condcovs.reserve(chunk_size);

            // Preallocate reusable H matrices
            if (local_data.H.empty()) local_data.H.resize(8);

            for (size_t i = range.begin(); i != range.end(); ++i) {
                const auto& [interp_state, border_states] = interp_to_borders_vec_[i];
                const auto& [left, right] = border_states;

                // Use cached poses/velocities
                const auto state_left = TimestampedPoseVelocity<PoseType>(
                    border_pose_cache[left.pose], border_vel_cache[left.vel], left.time);
                const auto state_right = TimestampedPoseVelocity<PoseType>(
                    border_pose_cache[right.pose], border_vel_cache[right.vel], right.time);

                PoseVelocity<PoseType> result;
                if (InterpJacobians) {
                    result = interpolator_.interpolatePoseAndVelocity(
                        state_left, state_right, interp_state.time, &local_data.H);

                    // Store jacobians
                    local_data.jacobians.emplace_back(interp_state.pose, left.pose, std::move(local_data.H[0]));
                    local_data.jacobians.emplace_back(interp_state.pose, left.vel, std::move(local_data.H[1]));
                    local_data.jacobians.emplace_back(interp_state.pose, right.pose, std::move(local_data.H[2]));
                    local_data.jacobians.emplace_back(interp_state.pose, right.vel, std::move(local_data.H[3]));
                    local_data.jacobians.emplace_back(interp_state.vel, left.pose, std::move(local_data.H[4]));
                    local_data.jacobians.emplace_back(interp_state.vel, left.vel, std::move(local_data.H[5]));
                    local_data.jacobians.emplace_back(interp_state.vel, right.pose, std::move(local_data.H[6]));
                    local_data.jacobians.emplace_back(interp_state.vel, right.vel, std::move(local_data.H[7]));
                } else {
                    result = interpolator_.interpolatePoseAndVelocity(
                        state_left, state_right, interp_state.time);
                }

                // Store results in thread-local vectors
                local_data.poses.emplace_back(interp_state.pose, std::move(result.pose));
                local_data.vels.emplace_back(interp_state.vel, std::move(result.vel));

                // Conditional covariance
                if (InterpCondCovs) {
                    auto state_tau = TimestampedPoseVelocity<PoseType>(result, interp_state.time);
                    Matrix2N Sigma_tau = interpolator_.computeConditionalCov(state_left, state_right, state_tau);
                    local_data.condcovs.emplace_back(interp_state, std::move(Sigma_tau));
                }
            }
        });

    // --- Step 4: Sequential merge ---
    Values values_interp;

    if (InterpJacobians) InterpJacobians->reserve(interp_to_borders_vec_.size());
    if (InterpCondCovs) InterpCondCovs->reserve(interp_to_borders_vec_.size());

    for (const auto& local_data : thread_data) {
        for (const auto& [key, pose] : local_data.poses) values_interp.insert(key, pose);
        for (const auto& [key, vel] : local_data.vels) values_interp.insert(key, vel);

        if (InterpJacobians) {
            for (const auto& [outer_key, inner_key, matrix] : local_data.jacobians) {
                auto& inner_map = (*InterpJacobians)[outer_key];
                if (inner_map.empty()) inner_map.reserve(8);
                inner_map[inner_key] = matrix;
            }
        }

        if (InterpCondCovs) {
            for (const auto& [state, cov] : local_data.condcovs)
                (*InterpCondCovs)[state] = cov;
        }
    }

    return values_interp;

#else
  // -------- Sequential optimized version (no TBB) --------
  // Goal: minimize repeated hash/map lookups and per-iteration allocations.
  // Strategy:
  //  * Pre-cache unique pose & velocity keys from boundary states.
  //  * Reuse a single Jacobian work vector across iterations.
  //  * Reduce operator[] nesting when filling InterpJacobians.

  Values values_interp; // result container

  // 1. Cache the actual pose/velocity objects (copy once, reuse many).
  //    (If PoseType or VelocityType are lightweight this still saves map/hashing cost.)
  // Preallocate space for border_pose_cache and border_vel_cache

  unordered_map<Key, PoseType>    border_pose_cache;
  unordered_map<Key, VelocityType> border_vel_cache;
  border_pose_cache.reserve(border_pose_keys_.size());
  border_vel_cache.reserve(border_vel_keys_.size());
  for (Key k : border_pose_keys_) border_pose_cache.emplace(k, values.at<PoseType>(k));
  for (Key k : border_vel_keys_)  border_vel_cache.emplace(k, values.at<VelocityType>(k));

  // 2. Reusable Jacobian storage (allocated once). The interpolate function overwrites entries.
  vector<Matrix> H(8);

  // 3. Iterate interpolation entries.
  for (const auto& kv : interp_to_borders_map_) {
    const auto& interp_state  = kv.first;
    const auto& border_states = kv.second;
    const auto& left  = border_states.first;
    const auto& right = border_states.second;

    const auto state_left  = TimestampedPoseVelocity<PoseType>(
        border_pose_cache[left.pose], border_vel_cache[left.vel], left.time);
    const auto state_right = TimestampedPoseVelocity<PoseType>(
        border_pose_cache[right.pose], border_vel_cache[right.vel], right.time);

    // 4. Interpolate (with or without Jacobians).
    PoseVelocity<PoseType> result;
    if (InterpJacobians) {
      result = interpolator_.interpolatePoseAndVelocity(state_left, state_right, interp_state.time, &H);
    } else {
      result = interpolator_.interpolatePoseAndVelocity(state_left, state_right, interp_state.time);
    }

    // 5. Insert interpolated pose & velocity.
    values_interp.insert(interp_state.pose, result.pose);
    values_interp.insert(interp_state.vel,  result.vel);

    // 6. Store Jacobians (minimize repeated map lookups by binding references once).
    if (InterpJacobians) {
      auto& J_pose = (*InterpJacobians)[interp_state.pose];
      auto& J_vel  = (*InterpJacobians)[interp_state.vel];
      J_pose[left.pose]   = H[0];
      J_pose[left.vel]    = H[1];
      J_pose[right.pose]  = H[2];
      J_pose[right.vel]   = H[3];
      J_vel[left.pose]    = H[4];
      J_vel[left.vel]     = H[5];
      J_vel[right.pose]   = H[6];
      J_vel[right.vel]    = H[7];
    }

    // 7. Conditional covariance if requested.
    if (InterpCondCovs) {
      auto state_tau = TimestampedPoseVelocity<PoseType>(result, interp_state.time);
      Matrix2N Sigma_tau = interpolator_.computeConditionalCov(state_left, state_right, state_tau);
      (*InterpCondCovs)[interp_state] = std::move(Sigma_tau);
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
