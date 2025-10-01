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
#include <gtsam/nonlinear/Values.h>

namespace gtsam {

// Simple, non-templated Gaussian fusion in a common tangent space
struct FusedGaussian {
  Vector mean;        // fused mean
  Matrix covariance;  // fused covariance
};

inline FusedGaussian FuseGaussians(const Vector& mu1, const Matrix& S1,
                                   const Vector& mu2, const Matrix& S2) {
  const Matrix iS1 = S1.inverse();
  const Matrix iS2 = S2.inverse();
  const Matrix S = (iS1 + iS2).inverse();
  const Vector m = S * (iS1 * mu1 + iS2 * mu2);
  return {m, S};
}

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
  NonlinearDensity(Key key, const VALUE& origin, const SharedNoiseModel& model,
                   const std::optional<Vector>& mean = {})
      : Base(key, origin, model, mean) {}

  /// @}
  /// @name Standard Destructor
  /// @{
  ~NonlinearDensity() override {}
  /// @}
  /// @name Testable
  /// @{

  /// print
  void print(const std::string& s, const KeyFormatter& keyFormatter =
                                       DefaultKeyFormatter) const override {
    std::cout << s << "NonlinearDensity on " << keyFormatter(this->key())
              << "\n";
    traits<T>::Print(this->origin_, "  origin: ");
    if (this->mean_) gtsam::print(*this->mean_, "  tangent space mean: ");
    if (this->noiseModel_)
      this->noiseModel_->print("  noise model: ");
    else
      std::cout << "no noise model\n";
  }

  /// equals
  bool equals(const NonlinearFactor& expected,
              double tol = 1e-9) const override {
    const auto* e = dynamic_cast<const NonlinearDensity*>(&expected);
    return e && Base::equals(*e, tol);
  }

  /// @}
  /// @name Standard API
  /// @{

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
   * Log-probability overload taking a Values container. This mirrors the
   * linear GaussianConditional interface so densities can be queried in a
   * uniform way when only a Values is available.
   */
  double logProbability(const Values& values) const {
    const T& x = values.at<T>(this->key());
    return logProbability(x);
  }

  /// Evaluate density P(x) using a Values container.
  double evaluate(const Values& values) const {
    const T& x = values.at<T>(this->key());
    return evaluate(x);
  }

  /**
   * Fusion operator implementing the (approximate) three-step Fusion
   * method in Ge–van Goor–Mahony (2024): choose a reference, express both
   * densities as extended concentrated Gaussians in that chart, fuse the
   * Gaussians, then reset to a zero-mean concentrated Gaussian.
   *
   * Notes/assumptions:
   *  - Only supports Gaussian noise models; throws otherwise.
   *  - Uses a full first-order Jacobian for the change of coordinates between
   * charts via the chain rule: J_map = (∂Local(x̂,q)/∂q)|_{q=origin} ·
   * (∂Retract(origin,δ)/∂δ)|_{δ=0}.
   *  - If both inputs share the same origin and mean, this reduces to
   *    classical Gaussian fusion: Σ⁺ = (Σ₁^{-1}+Σ₂^{-1})^{-1} at the same
   * origin.
   */
  NonlinearDensity operator*(const NonlinearDensity& other) const {
    // 0) Sanity checks
    if (this->key() != other.key())
      throw std::invalid_argument("NonlinearDensity::operator*: keys differ");

    // Extract Gaussian noise models and covariances
    auto g1 =
        std::dynamic_pointer_cast<noiseModel::Gaussian>(this->noiseModel());
    auto g2 =
        std::dynamic_pointer_cast<noiseModel::Gaussian>(other.noiseModel());
    if (!g1 || !g2)
      throw std::runtime_error(
          "NonlinearDensity::operator*: only Gaussian noise models are "
          "supported");

    const Matrix Sigma1 = g1->covariance();
    const Matrix Sigma2 = g2->covariance();

    // 1) Reference selection (info-weighted in identity chart)
    const Vector mu1_ref = traits<T>::Logmap(this->origin_);
    const Vector mu2_ref = traits<T>::Logmap(other.origin_);
    const Matrix SigRef = (Sigma1.inverse() + Sigma2.inverse()).inverse();
    const Vector muRef =
        SigRef * (Sigma1.inverse() * mu1_ref + Sigma2.inverse() * mu2_ref);
    const T xhat = traits<T>::Expmap(muRef);

    // 2) Map both ECGs to chart at x̂
    Vector mu1_hat, mu2_hat;
    Matrix S1_hat, S2_hat;
    mapToReference_(xhat, this->origin_, this->mean_, Sigma1, mu1_hat, S1_hat);
    mapToReference_(xhat, other.origin_, other.mean_, Sigma2, mu2_hat, S2_hat);

    // 3) Classical Gaussian fusion in common tangent
    const FusedGaussian fg = FuseGaussians(mu1_hat, S1_hat, mu2_hat, S2_hat);

    // 4) Reset to zero-mean at fused origin
    const T xplus = traits<T>::Retract(xhat, fg.mean);
    const SharedNoiseModel modelPlus =
        noiseModel::Gaussian::Covariance(fg.covariance);
    return NonlinearDensity(this->key(), xplus, modelPlus);
  }

  /** Reset this ECG to a new reference x̂: map (origin, mean, Σ) into the
   *  chart at x̂ via first-order Jacobians, then retract by the mapped mean
   *  so the result has zero mean at the new origin.
   *  Requires a Gaussian noise model; throws otherwise. */
  NonlinearDensity resetToZeroMeanAt(const T& xhat) const {
    auto g =
        std::dynamic_pointer_cast<noiseModel::Gaussian>(this->noiseModel());
    if (!g)
      throw std::runtime_error(
          "NonlinearDensity::resetToZeroMeanAt: only Gaussian noise models are "
          "supported");

    const Matrix& Sigma = g->covariance();

    Vector mu_hat;
    Matrix S_hat;
    mapToReference_(xhat, this->origin_, this->mean_, Sigma, mu_hat, S_hat);

    const T xplus = traits<T>::Retract(xhat, mu_hat);
    const SharedNoiseModel modelPlus = noiseModel::Gaussian::Covariance(S_hat);
    return NonlinearDensity(this->key(), xplus, modelPlus);
  }

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
  /// @}

 private:
  static void mapToReference_(const T& xhat, const T& origin,
                              const std::optional<Vector>& mean,
                              const Matrix& Sigma, Vector& mu_hat,
                              Matrix& S_hat) {
    const size_t n = Sigma.rows();
    Matrix Hlp, Hlq;  // d Local(xhat,q)/d p and d Local(xhat,q)/d q
    Vector r = traits<T>::Local(xhat, origin, Hlp, Hlq);

    Matrix Hr_p, Hr_v;  // d Retract(origin,δ)/d origin and /d δ at δ=0
    traits<T>::Retract(origin, Vector::Zero(n), Hr_p, Hr_v);

    const Matrix Jmap = Hlq * Hr_v;  // chain rule
    const Vector m = mean.value_or(Vector::Zero(n));

    mu_hat = r + Jmap * m;
    S_hat = Jmap * Sigma * Jmap.transpose();
  }

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
