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
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>


#include <vector>

using namespace std;

namespace gtsam {

/* @brief struct for keeping track interpolation information*/
struct InterpData {
  array<Key, 2> state_keys;
  array<Key, 2> vel_keys;
  array<double, 2> times;
  double interp_time;
};

/* Wrapper class that allows a variable of a factor to be replaced by a WNOA
 * interpolation*/
// the iterpolation index specifies which pair of values are used to interpolate
// a new value
template <class InnerFactorType, class InterpType, int InterpIndex>
class WNOAInterpFactor : public NoiseModelFactor {
 private:
  using Base = NoiseModelFactor;
  using This = WNOAInterpFactor<InnerFactorType, InterpType, InterpIndex>;

  const InnerFactorType inner_factor_;

  static inline KeyVector get_outer_keys(const InnerFactorType& inner_factor,
                                  const InterpData& interp_data) {
    // // retreive inner factor keys and interpolation keys
    // KeyVector inner_keys = inner_factor.keys();
    // KeyVector interp_keys = {interp_data.state_keys[0],
    // interp_data.vel_keys[0],
    //                          interp_data.state_keys[1],
    //                          interp_data.vel_keys[1]};
    // // remove interpolated variable key
    // inner_keys.erase(InterpIndex);
    // // return combined keys with interpolated keys first.
    // return interp_keys.insert(interp_keys.end(), inner_keys.begin(),
    // inner_keys.end());
    return inner_factor.keys();
  }

 public:
  WNOAInterpFactor(const InnerFactorType& inner_factor, InterpData& interp_data)
      : Base(inner_factor.noiseModel(),
             This::get_outer_keys(inner_factor, interp_data)),
        inner_factor_(inner_factor) {};

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
    //Interpolation goes here

    // Call inner factor's evaluate function with updated values.
    auto error = inner_factor_.unwhitenedError(values, H);

    //Jacobian computations

    return error
  }
};


}  // namespace gtsam