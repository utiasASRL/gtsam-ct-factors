/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file LocBearingRangeFactor.h
 * @date Apr 1, 2010
 * @author Connor Holmes
 * @brief a single factor contains both the bearing and the range to prevent
 * handle to pair bearing and range factors. This version of the factor assumes
 * that the second input is a fixed landmark, to allow localization.
 */

#pragma once

#include <gtsam/geometry/BearingRange.h>
#include <gtsam/nonlinear/ExpressionFactor.h>

namespace gtsam {

/**
 * Binary factor for a bearing/range measurement
 * @ingroup sam
 */
template <typename A1, typename A2,
          typename B = typename Bearing<A1, A2>::result_type,
          typename R = typename Range<A1, A2>::result_type>
class LocBearingRangeFactor
    : public ExpressionFactorN<BearingRange<A1, A2>, A1, A2> {
 private:
  typedef BearingRange<A1, A2> T;
  typedef ExpressionFactorN<T, A1, A2> Base;
  typedef LocBearingRangeFactor<A1, A2> This;
  A2 landmark;

 public:
  typedef std::shared_ptr<This> shared_ptr;

  /// Default constructor
  LocBearingRangeFactor() {}

  /// Construct from BearingRange instance
  LocBearingRangeFactor(Key key1, const A2& landmark, const T& bearingRange,
                        const SharedNoiseModel& model)
      : Base({{key1}}, model, T(bearingRange)), landmark(landmark) {
    this->initialize(expression({{key1}}));
  }

  /// Construct from separate bearing and range
  LocBearingRangeFactor(Key key1, const A2& landmark, const B& measuredBearing,
                        const R& measuredRange, const SharedNoiseModel& model)
      : Base({{key1}}, model, T(measuredBearing, measuredRange)),
        landmark(landmark) {
    this->initialize(expression({{key1}}));
  }

  ~LocBearingRangeFactor() override {}

  /// @return a deep copy of this factor
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  // Return measurement expression
  Expression<T> expression(
      const typename Base::ArrayNKeys& keys) const override {
    return Expression<T>(T::Measure, Expression<A1>(keys[0]),
                         Expression<A2>(landmark));
  }

  Vector evaluateError(const A1& a1, const A2& a2,
                       OptionalMatrixType H1 = OptionalNone,
                       OptionalMatrixType H2 = OptionalNone) const {
    std::vector<Matrix> Hs(2);
    const auto& keys = Factor::keys();
    const Vector error = this->unwhitenedError(
        {{keys[0], genericValue(a1)}}, Hs);
    if (H1) *H1 = Hs[0];
    if (H2) *H2 = Hs[1];
    return error;
  }

  /// print
  void print(const std::string& s = "",
             const KeyFormatter& kf = DefaultKeyFormatter) const override {
    std::cout << s << "LocBearingRangeFactor" << std::endl;
    Base::print(s, kf);
  }

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& boost::serialization::make_nvp(
        "Base", boost::serialization::base_object<Base>(*this));
  }
#endif
};  // LocBearingRangeFactor

/// traits
template <typename A1, typename A2, typename B, typename R>
struct traits<LocBearingRangeFactor<A1, A2, B, R> >
    : public Testable<LocBearingRangeFactor<A1, A2, B, R> > {};

}  // namespace gtsam
