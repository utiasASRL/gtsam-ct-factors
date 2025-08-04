#include <gtsam/base/Lie.h>
#include <gtsam/base/VectorSpace.h>
#include <gtsam/base/ProductLieGroup.h>
#include <gtsam/base/Testable.h>
#include <gtsam/geometry/Point1.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <cassert>

#pragma once
using namespace std;

namespace gtsam {

template <class Pose>
class WNOAMotionFactor
    : public NoiseModelFactorN<Pose, typename traits<Pose>::TangentVector, Pose,
                               typename traits<Pose>::TangentVector> {
  // Check that Pose type is a testable Lie group
  GTSAM_CONCEPT_ASSERT(IsTestable<Pose>);
  // We currently support vector spaces and Lie groups
  // SL: Can we use GTSAM_CONCEPT_ASSERT here?
  static_assert(std::is_same_v<typename traits<Pose>::structure_category, lie_group_tag> ||
                std::is_same_v<typename traits<Pose>::structure_category, vector_space_tag>,
                "Pose type must be either a Lie group or vector space");
  
  GTSAM_CONCEPT_ASSERT(IsLieGroup<Pose>);  // (CH) this could potentially be
                                           // changed to Manifold check
 public:
  static constexpr int dim = traits<Pose>::dimension;

 private:
  // expose tangent vector
  using Velocity = typename gtsam::traits<Pose>::TangentVector;
  using MatrixN = Eigen::Matrix<double, dim, dim>;
  using VectorN = Eigen::Matrix<double, dim, 1>;
  using Matrix2N = Eigen::Matrix<double, 2*dim, 2*dim>;
  using Vector2N = Eigen::Matrix<double, 2*dim, 1>;
  using MatrixNx2N = Eigen::Matrix<double, dim, 2*dim>;
  typedef NoiseModelFactorN<Pose, Velocity, Pose, Velocity> Base;
  typedef WNOAMotionFactor This;
  double delta_t_;

 public:
  // Provide access to the Matrix& version of evaluateError:
  using Base::evaluateError;
  // Dimension variable, used for convenience
  /**
  Constructor
  @param key1 key for the pose at t_k
  @param key2 key for the velocity at t_k
  @param key3 key for the pose at t_{k+1}
  @param key4 key for the velocity at t_{k+1}
  @param delta_t Magnitude of the timestep, t_{k+1}-t_k, used to construct an
  @param Q Input noise matrix, Noise model is constructed from this and delta_t
  altered noise model based on the provided model.
  */
  WNOAMotionFactor(Key key1, Key key2, Key key3, Key key4, const double delta_t,
                   const VectorN& Q)
      : Base(This::buildWNOANoiseModel(delta_t, Q), key1, key2, key3, key4),
        delta_t_(delta_t) {}

  ~WNOAMotionFactor() override {}
  /** implement functions needed for Testable */
  /** print */
  void print(
      const std::string& s = "",
      const KeyFormatter& keyFormatter = DefaultKeyFormatter) const override {
    std::cout << s << "WNOAMotionFactor(" << keyFormatter(this->key1()) << ","
              << keyFormatter(this->key2()) << "," << keyFormatter(this->key3())
              << "," << keyFormatter(this->key4()) << ")\n";
    this->noiseModel_->print("  noise model: ");
  }

  /** equals */
  bool equals(const NonlinearFactor& expected,
              double tol = 1e-9) const override {
    const This* e = dynamic_cast<const This*>(&expected);
    return e != nullptr && Base::equals(*e, tol);
  }

  /** functions required to be a factor */
  Vector evaluateError(const Pose& p1, const Velocity& v1, const Pose& p2,
                       const Velocity& v2, OptionalMatrixType Hp1,
                       OptionalMatrixType Hv1, OptionalMatrixType Hp2,
                       OptionalMatrixType Hv2) const override {
    // Note that p1 = T(t_k), p2 = T(t_{k+1})
    //  compute xi = log(T_k^-1 T_{k+1})^check
    Matrix dxi_dT1(dim, dim);
    Matrix dxi_dT2(dim, dim);
    Vector xi(dim);
    Matrix right_jac_inv(dim,dim);
    if (Hp1 || Hp2) {
      Matrix dbetween_p1;
      Matrix dbetween_p2;
      xi = traits<Pose>::Logmap(traits<Pose>::Between(p1, p2, &dbetween_p1, &dbetween_p2), &right_jac_inv);
      dxi_dT1 = right_jac_inv * dbetween_p1;
      dxi_dT2 = right_jac_inv * dbetween_p2;
    } else {
      xi = traits<Pose>::Logmap(traits<Pose>::Between(p1, p2), &right_jac_inv);
    }
    // Compute error
    Vector err(2 * dim);

    err << xi - delta_t_ * v1, right_jac_inv * v2 - v1;
    
    // Derivative of velocity error wrt xi
    Matrix dvErr_dxi(dim, dim);
    if (Hp1 || Hp2) {
      // Derivative of velocity error wrt xi
      // Zero for vector spaces, use an approximation for Lie groups
      if constexpr (std::is_same_v<typename traits<Pose>::structure_category, vector_space_tag>) {
        dvErr_dxi.setZero();
      } else {
        // For Lie groups
        dvErr_dxi = -Pose::adjointMap(v2) / 2.0 -
          (Pose::adjointMap(Pose::adjointMap(xi) * v2) +
           Pose::adjointMap(xi) * Pose::adjointMap(v2)) /
              12.0;
      }
    }
    // Compute Final Jacobians
    if (Hp1) {
      // Derivative of error wrt pose p1
      *Hp1 = (Matrix(2 * dim, dim) << dxi_dT1, dvErr_dxi * dxi_dT1).finished();
    }
    if (Hv1) {
      // Derivative of error wrt velocity v1
      Eigen::Matrix<double, dim, dim> I =
          Eigen::Matrix<double, dim, dim>::Identity();
      *Hv1 = (Matrix(2 * dim, dim) << -delta_t_ * I, -I).finished();
    }
    if (Hp2) {
      // Derivative of error wrt pose p2
      *Hp2 = (Matrix(2 * dim, dim) << dxi_dT2, dvErr_dxi * dxi_dT2).finished();
    }
    if (Hv2) {
      // Derivative of error wrt velocity v2
      Eigen::Matrix<double, dim, dim> Zero =
          Eigen::Matrix<double, dim, dim>::Zero();
      *Hv2 = (Matrix(2 * dim, dim) << Zero, right_jac_inv).finished();
    }

    return err;
  }

  // Functions to build the specific covariance for the WNOA model.
  static Matrix2N buildWNOACovariance(double timestep, const VectorN& Q) {
    assert(Q.size() == dim && "WNOA Q-matrix Dimensions");  // todo (Daniel): shouldn't be necessary with static types
    // construct the covariance matrix for the WNOA factor
    Matrix2N covariance;
    MatrixN Q_diag = Q.asDiagonal();
    covariance << (1.0 / 3.0 * pow(timestep, 3)) * Q_diag,
        (1.0 / 2.0 * pow(timestep, 2)) * Q_diag, (1.0 / 2.0 * pow(timestep, 2)) * Q_diag,
        timestep * Q_diag;
    return covariance;
  }

  static inline noiseModel::Gaussian::shared_ptr buildWNOANoiseModel(
      double timestep, const VectorN& Q) {
    return noiseModel::Gaussian::Covariance(
        buildWNOACovariance(timestep, Q));
  }


  static Matrix2N transitionFunction(double delta_t) {
    // Construct the transition matrix for the WNOA factor
    Matrix2N F;
    F << Matrix::Identity(dim, dim), delta_t * Matrix::Identity(dim, dim),
        Matrix::Zero(dim, dim), Matrix::Identity(dim, dim);
    return F;
  }
};


// Make factors testable
template <class Pose>
struct traits<WNOAMotionFactor<Pose>>
    : public Testable<WNOAMotionFactor<Pose>> {};

// Explicit Instantiation.
template class WNOAMotionFactor<Point1>;
template class WNOAMotionFactor<Point2>;
template class WNOAMotionFactor<Point3>;
template class WNOAMotionFactor<Pose2>;
template class WNOAMotionFactor<Pose3>;

}  // namespace gtsam