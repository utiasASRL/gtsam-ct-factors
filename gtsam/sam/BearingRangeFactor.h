/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file BearingRangeFactor.h
 * @date Apr 1, 2010
 * @author Kai Ni
 * @brief a single factor contains both the bearing and the range to prevent
 * handle to pair bearing and range factors
 */

#pragma once

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/BearingRange.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>

#include <type_traits>

namespace gtsam {

class Unit3;

/**
 * Binary factor for a bearing/range measurement
 * @ingroup sam
 */
template <typename A1, typename A2,
          typename B = typename Bearing<A1, A2>::result_type,
          typename R = typename Range<A1, A2>::result_type>
class BearingRangeFactor : public NoiseModelFactorN<A1, A2> {
 private:
  typedef BearingRange<A1, A2> T;
  typedef NoiseModelFactorN<A1, A2> Base;
  typedef BearingRangeFactor<A1, A2, B, R> This;

 public:
  typedef std::shared_ptr<This> shared_ptr;

 private:
  T measured_;

 public:
  /// Default constructor
  BearingRangeFactor() = default;

  /// Construct from keys, a combined bearing-range measurement, and a noise
  /// model.
  BearingRangeFactor(Key key1, Key key2, const T& bearingRange,
                     const SharedNoiseModel& model)
      : Base(model, key1, key2), measured_(bearingRange) {}

  /// Construct from keys, separate bearing/range measurements, and a noise
  /// model.
  BearingRangeFactor(Key key1, Key key2, const B& measuredBearing,
                     const R& measuredRange, const SharedNoiseModel& model)
      : Base(model, key1, key2), measured_(measuredBearing, measuredRange) {}

  /// Destroy this factor.
  ~BearingRangeFactor() override {}

  /// @return a deep copy of this factor
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  /// @return the measurement
  const T& measured() const { return measured_; }

  /// Check equality up to a tolerance.
  bool equals(const NonlinearFactor& f, double tol = 1e-9) const override {
    const This* p = dynamic_cast<const This*>(&f);
    return p != nullptr && Base::equals(f, tol) &&
           traits<T>::Equals(measured_, p->measured_, tol);
  }

  /// Evaluate the unwhitened bearing-range error and optional Jacobians.
  Vector evaluateError(const A1& a1, const A2& a2,
                       OptionalMatrixType H1 = OptionalNone,
                       OptionalMatrixType H2 = OptionalNone) const override {
    constexpr int dimB = traits<B>::dimension;
    constexpr int dimR = traits<R>::dimension;
    constexpr int dim1 = traits<A1>::dimension;
    constexpr int dim2 = traits<A2>::dimension;

    typename MakeJacobian<B, A1>::type HB1;
    typename MakeJacobian<B, A2>::type HB2;
    typename MakeJacobian<R, A1>::type HR1;
    typename MakeJacobian<R, A2>::type HR2;
    typename traits<R>::ChartJacobian::Jacobian HlocalR;

    const B predictedBearing =
        Bearing<A1, A2>()(a1, a2, H1 ? &HB1 : nullptr, H2 ? &HB2 : nullptr);
    const R predictedRange =
        Range<A1, A2>()(a1, a2, H1 ? &HR1 : nullptr, H2 ? &HR2 : nullptr);

    Vector error(dimB + dimR);
    Matrix HerrorBearing;
    if constexpr (std::is_same_v<B, Unit3>) {
      const auto localBearingError = [&](const B& value) -> Vector {
        return -traits<B>::Local(value, measured_.bearing());
      };
      error.head(dimB) = localBearingError(predictedBearing);
      if (H1 || H2) {
        HerrorBearing = numericalDerivative11<Vector, B>(localBearingError,
                                                         predictedBearing);
      }
    } else {
      typename traits<B>::ChartJacobian::Jacobian HlocalB;
      error.head(dimB) =
          -traits<B>::Local(predictedBearing, measured_.bearing(),
                            (H1 || H2) ? &HlocalB : nullptr, OptionalNone);
      if (H1 || H2) {
        HerrorBearing = -HlocalB;
      }
    }
    error.tail(dimR) =
        -traits<R>::Local(predictedRange, measured_.range(),
                          (H1 || H2) ? &HlocalR : nullptr, OptionalNone);

    if (H1) {
      H1->resize(dimB + dimR, dim1);
      H1->topRows(dimB) = HerrorBearing * HB1;
      H1->bottomRows(dimR) = -HlocalR * HR1;
    }
    if (H2) {
      H2->resize(dimB + dimR, dim2);
      H2->topRows(dimB) = HerrorBearing * HB2;
      H2->bottomRows(dimR) = -HlocalR * HR2;
    }
    return error;
  }

  /// print
  void print(const std::string& s = "",
             const KeyFormatter& kf = DefaultKeyFormatter) const override {
    std::cout << s << "BearingRangeFactor" << std::endl;
    Base::print(s, kf);
    traits<T>::Print(measured_, "  measured: ");
  }

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  friend class boost::serialization::access;
  /// Serialize this factor.
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& boost::serialization::make_nvp(
        "NoiseModelFactor2", boost::serialization::base_object<Base>(*this));
    ar& BOOST_SERIALIZATION_NVP(measured_);
  }
#endif
};  // BearingRangeFactor

/// traits
template <typename A1, typename A2, typename B, typename R>
struct traits<BearingRangeFactor<A1, A2, B, R> >
    : public Testable<BearingRangeFactor<A1, A2, B, R> > {};

}  // namespace gtsam
