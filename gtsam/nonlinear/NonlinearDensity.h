/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file NonlinearDensity.h
 * @date September 30, 2025
 * @author Frank Dellaert
 * @brief A nonlinear density, inherits from NonlinearLikelihood.
 */

#pragma once

#include <gtsam/nonlinear/NonlinearLikelihood.h>

namespace gtsam {

/**
 * A nonlinear density, inherits from NonlinearLikelihood.
 * @ingroup nonlinear
 */
template <class VALUE>
class NonlinearDensity : public NonlinearLikelihood<VALUE> {
 public:
  typedef VALUE T;
  typedef NonlinearLikelihood<VALUE> Base;

  /// @name Standard Constructors
  /// @{

  /// Default constructor for serialization.
  NonlinearDensity() {}

  /// Constructor
  NonlinearDensity(Key key, const VALUE& origin, const SharedNoiseModel& model)
      : Base(key, origin, model) {}

  /// @}
  /// @name Standard Destructor
  /// @{
  ~NonlinearDensity() override {}
  /// @}

  /**
   * Calculate the log-probability of the given value.
   * error(x) as defined for a GTSAM factor already equals 0.5 * ||r(x)||^2_Σ
   * (i.e. the negative log-likelihood without the normalization constant).
   * Hence: log P(x) = log k - error(x).
   */
  double logProbability(const T& x) const {
    return -(negLogConstant() + this->error(x));
  }

  /**
   * Evaluate the probability density at the given value.
   * P(x) = exp(logProbability(x)).
   */
  double evaluate(const T& x) const { return exp(logProbability(x)); }

  /**
   * Calculate the normalization constant for the density.
   * For a Gaussian noise model with covariance Σ, we return
   *   - log k = 0.5 * n * log(2*pi) + 0.5 * log |Σ|
   * where n = dim().  Note: gaussian->logDeterminant() returns log|Σ|.
   * For non-Gaussian noise models this is not (straightforwardly) defined and
   * we throw.
   */
  double negLogConstant() const {
    // Get number of rows
    const size_t n = this->dim();

    // Get noise model and dynamic cast to Gaussian
    const auto& noiseModel = this->noiseModel();
    auto gaussian = std::dynamic_pointer_cast<noiseModel::Gaussian>(noiseModel);
    if (gaussian) {
      constexpr double log2pi = 1.8378770664093454835606594728112;  // log(2*pi)
      const double logDetSigma = gaussian->logDeterminant();        // log |Σ|
      return 0.5 * n * log2pi + 0.5 * logDetSigma;
    }

    // If not Gaussian, throw an error
    throw std::runtime_error(
        "NonlinearDensity::negLogConstant() is only implemented for "
        "Gaussian noise models. The noise model used is of type " +
        std::string(typeid(*noiseModel).name()));
  }

 private:
#ifdef GTSAM_ENABLE_BOOST_SERIALIZATION
  /** Serialization function */
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    ar& boost::serialization::make_nvp(
        "NonlinearLikelihood", boost::serialization::base_object<Base>(*this));
  }
#endif
};

}  // namespace gtsam
