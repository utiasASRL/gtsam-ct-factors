#pragma once

#include <gtsam/base/Lie.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/VectorSpace.h>
#include <gtsam/geometry/Point1.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/Interpolator.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;

namespace gtsam {

/* @brief State data structure for keeping track of pose and velocity keys as
 * well as associated timestamp. Used in GP interpolation.*/
struct StateData {
  Key pose;
  Key vel;
  double time;
  // Default constructor
  StateData() = default;
  // Constructor
  StateData(Key pose_in, Key vel_in, double time_in)
      : pose(pose_in), vel(vel_in), time(time_in) {};
};

/* @brief Struct for keeping track GP interpolation information.
 * Assumes that the times of the interpolated states are within the border state
 * times*/
struct InterpData {
  // Border state information
  StateData border_states[2];
  // Interpolated state information
  vector<StateData> interp_states;
  // Power Spectral Density Matrix
  Vector Q_psd;
};

/* Wrapper class that allows a variable of a factor to be replaced by a WNOA
 * interpolation */
template <class InnerFactorType, class PoseType>
class WNOAInterpFactor : public NoiseModelFactor {
 private:
  using Base = NoiseModelFactor;
  using This = WNOAInterpFactor<InnerFactorType, PoseType>;
  using VelocityType = typename gtsam::traits<PoseType>::TangentVector;
  static constexpr int dim = traits<PoseType>::dimension;
  // Inner factor that is called on interpolated values
  const InnerFactorType inner_factor_;
  // Interpolator class
  const Interpolator<PoseType> interpolator_;
  // Store interpolation information
  const InterpData interp_data_;

 public:
  WNOAInterpFactor(const InnerFactorType& inner_factor,
                   const InterpData& interp_data,
                   const Eigen::Vector<double, dim>& Q_psd)
      : Base(inner_factor.noiseModel(),
             This::get_outer_keys(inner_factor.keys(), interp_data)),
        inner_factor_(inner_factor),
        interpolator_(Q_psd),
        interp_data_(interp_data) {};

  ~WNOAInterpFactor() override {};

  /** implement functions needed for Testable */
  /** print */
  void print(
      const std::string& s = "",
      const KeyFormatter& keyFormatter = DefaultKeyFormatter) const override {
    cout << s << "WNOAInterpFactor on ";
    for (const auto& k : this->keys()) {
      cout << keyFormatter(k) << " ";  // raw numeric key
    }
    cout << endl;
    this->inner_factor_.print("Inner Factor: ");
  }
  /** equals */
  bool equals(const NonlinearFactor& expected,
              double tol = 1e-9) const override {
    const This* e = dynamic_cast<const This*>(&expected);
    return e != nullptr && Base::equals(*e, tol);
  }

  // Provide access to the Matrix& and the non-Jacobian version of
  // unwhitenedError
  using Base::unwhitenedError;

  // Override of evaluateError required by NoiseModelFactorN derivatives
  Vector unwhitenedError(const Values& values,
                         OptionalMatrixVecType H = nullptr) const override {
    // process interpolated states
    const Values values_interp = get_interp_values(values);

    // construct values for inner factor
    Values values_inner;
    for (Key key : inner_factor_.keys()) {
      if (values_interp.exists(key)) {
        // found key in interpolated values
        values_inner.insert(key, values_interp.at(key));
      } else if (values.exists(key)) {
        // found key in values passed to outer factor
        values_inner.insert(key, values.at(key));
      } else {
        // if key is not found then there is a problem
        throw runtime_error("Key " + DefaultKeyFormatter(key) +
                            " not in interpolated states or outer keys");
      }
    }

    // Call inner factor's evaluate function with updated values.
    auto error = inner_factor_.unwhitenedError(values_inner, H);

    // Jacobian computations

    return error;
  }

 private:
  /* @brief Compute the interpolated values based on border state values */
  Values get_interp_values(const Values& values) const {
    // retrieve bordering states
    const auto state_vel_0 =
        make_pair(values.at<PoseType>(interp_data_.border_states[0].pose),
                  values.at<VelocityType>(interp_data_.border_states[0].vel));
    const auto state_vel_1 =
        make_pair(values.at<PoseType>(interp_data_.border_states[1].pose),
                  values.at<VelocityType>(interp_data_.border_states[1].vel));
    // process all interpolated states, put results in values dictionary
    Values values_interp;
    for (StateData interp_state : interp_data_.interp_states) {
      // Get interpolated state velocity pair
      auto state_vel_pair = interpolator_.interpolatePoseAndVelocity(
          state_vel_0, interp_data_.border_states[0].time, state_vel_1,
          interp_data_.border_states[1].time, interp_state.time);
      // insert into values structure
      values_interp.insert(interp_state.pose, state_vel_pair.first);
      values_interp.insert(interp_state.vel, state_vel_pair.second);
    }
    return values_interp;
  }

  /* @brief This function retrieves the approapriate keys for the outer factor
   * based on the interpolation data and the inner factor*/
  static inline KeyVector get_outer_keys(const KeyVector& inner_keys,
                                         const InterpData& interp_data) {
    // retreive inner factor keys and interpolation keys
    KeyVector new_keys = {
        interp_data.border_states[0].pose,
        interp_data.border_states[0].vel,
        interp_data.border_states[1].pose,
        interp_data.border_states[1].vel,
    };
    // get set of interpolated keys
    unordered_set<Key> interp_key_set;
    for (StateData interp_state_key : interp_data.interp_states) {
      interp_key_set.insert(interp_state_key.pose);
      interp_key_set.insert(interp_state_key.vel);
    }
    // add inner keys that are not being interpolated
    for (Key key : inner_keys) {
      if (interp_key_set.count(key) == 0) {
        new_keys.push_back(key);
      }
    }

    // return the modified set of keys
    return new_keys;
  }
};

}  // namespace gtsam