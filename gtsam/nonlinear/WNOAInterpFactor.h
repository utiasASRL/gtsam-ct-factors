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

#include <utility>
#include <vector>

using namespace std;

namespace gtsam {

/* @brief struct for keeping track interpolation information*/
struct InterpData {
  array<Key, 2> state_keys;
  array<Key, 2> vel_keys;
  Key interp_key;
  array<double, 2> times;
  double interp_time;
};

/* Wrapper class that allows a variable of a factor to be replaced by a WNOA
 * interpolation*/
template <class InnerFactorType, class PoseType>
class WNOAInterpFactor : public NoiseModelFactor {
 private:
  using Base = NoiseModelFactor;
  using This = WNOAInterpFactor<InnerFactorType, PoseType> using VelocityType =
      typename gtsam::traits<PoseType>::TangentVector;
  // Inner factor that is called on interpolated values
  const InnerFactorType inner_factor_;
  // Interpolator class
  const Interpolator<PoseType> interp_;
  // Store interpolation information
  const InterpData interp_data_;

 public:
  WNOAInterpFactor(const InnerFactorType& inner_factor, InterpData& interp_data,
                   const Vector& Q_psd)
      : Base(inner_factor.noiseModel(),
             This::get_outer_keys(inner_factor, interp_data)),
        inner_factor_(inner_factor),
        interp_(Q_psd),
        interp_data_(interp_data) {};

  ~WNOAInterpFactor() override {};

  /** implement functions needed for Testable */
  /** print */
  void print(
      const std::string& s = "",
      const KeyFormatter& keyFormatter = DefaultKeyFormatter) const override {
    this->inner_factor_.print("WNOA Interpolation Wrapper:");
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
    // Interpolate the value
    auto state_vel_0 =
        make_pair(values.at<PoseType>(interp_data_.state_keys[0]),
                  values.at<VelocityType>(interp_data_.vel_keys[0]));
    auto state_vel_1 =
        make_pair(values.at<PoseType>(interp_data_.state_keys[1]),
                  values.at<VelocityType>(interp_data_.vel_keys[1]));
    auto state_vel_interp = interp_.interpolatePoseAndVelocity(
        state_vel_0, interp_data_.times[0], state_vel_1, interp_data_.times[1],
        interp_data_.interp_time);
    // Create modified values input
    
    // Call inner factor's evaluate function with updated values.
    auto error = inner_factor_.unwhitenedError(values, H);

    // Jacobian computations

    return error;
  }

 private:
  /* @brief This function retrieves the approapriate keys for the outer factor
   * based on the interpolation data and the inner factor*/
  static inline KeyVector get_outer_keys(const KeyVector& inner_keys,
                                         const InterpData& interp_data) {
    // retreive inner factor keys and interpolation keys
    KeyVector new_keys = {interp_data.state_keys[0], interp_data.vel_keys[0],
                          interp_data.state_keys[1], interp_data.vel_keys[1]};
    // Add the other keys (but not the interpolated one)
    // NOTE: (CH) Not sure if this is the best way to do this. Compile time operation would be faster
    for (Key key : inner_keys) {
      if (key != interp_data.interp_key){
        new_keys.push_back(key);
      }
    }

    // return new_keys;
    return inner_keys;
  }
};

}  // namespace gtsam