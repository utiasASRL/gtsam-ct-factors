/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file   Similarity3.cpp
 * @brief  Implementation of Similarity3 transform
 * @author Paul Drews
 * @author John Lambert
 */

#include <gtsam/geometry/Similarity3.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Manifold.h>
#include <gtsam/slam/KarcherMeanFactor-inl.h>

namespace gtsam {

namespace internal {
/// Subtract centroids from point pairs.
static Point3Pairs subtractCentroids(const Point3Pairs &abPointPairs,
                                    const Point3Pair &centroids) {
  Point3Pairs d_abPointPairs;
  for (const auto& [a, b] : abPointPairs) {
    Point3 da = a - centroids.first;
    Point3 db = b - centroids.second;
    d_abPointPairs.emplace_back(da, db);
  }
  return d_abPointPairs;
}

/// Form inner products x and y and calculate scale.
// We force the scale to be a non-negative quantity
// (see Section 10.1 of https://ethaneade.com/lie_groups.pdf)
static double calculateScale(const Point3Pairs &d_abPointPairs,
                             const Rot3 &aRb) {
  double x = 0, y = 0;
  for (const auto& [da, db] : d_abPointPairs) {
    const Vector3 da_prime = aRb * db;
    y += da.transpose() * da_prime;
    x += da_prime.transpose() * da_prime;
  }
  const double s = std::fabs(y / x);
  return s;
}

/// Form outer product H.
static Matrix3 calculateH(const Point3Pairs &d_abPointPairs) {
  Matrix3 H = Z_3x3;
  for (const auto& [da, db] : d_abPointPairs) {
    H += da * db.transpose();
  }
  return H;
}

/// This method estimates the similarity transform from differences point pairs,
// given a known or estimated rotation and point centroids.
static Similarity3 align(const Point3Pairs &d_abPointPairs, const Rot3 &aRb,
                         const Point3Pair &centroids) {
  const double s = calculateScale(d_abPointPairs, aRb);
  // dividing aTb by s is required because the registration cost function
  // minimizes ||a - sRb - t||, whereas Sim(3) computes s(Rb + t)
  const Point3 aTb = (centroids.first - s * (aRb * centroids.second)) / s;
  return Similarity3(aRb, aTb, s);
}

/// This method estimates the similarity transform from point pairs, given a known or estimated rotation.
// Refer to: http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2005/Zinsser05-PSR.pdf Chapter 3
static Similarity3 alignGivenR(const Point3Pairs &abPointPairs,
                               const Rot3 &aRb) {
  auto centroids = means(abPointPairs);
  auto d_abPointPairs = internal::subtractCentroids(abPointPairs, centroids);
  return align(d_abPointPairs, aRb, centroids);
}
}  // namespace internal

Similarity3::Similarity3() :
    t_(0,0,0), s_(1) {
}

Similarity3::Similarity3(double s) :
    t_(0,0,0), s_(s) {
}

Similarity3::Similarity3(const Rot3& R, const Point3& t, double s) :
    R_(R), t_(t), s_(s) {
}

Similarity3::Similarity3(const Matrix3& R, const Vector3& t, double s) :
    R_(R), t_(t), s_(s) {
}

Similarity3::Similarity3(const Matrix4& T) :
    R_(T.topLeftCorner<3, 3>()), t_(T.topRightCorner<3, 1>()), s_(1.0 / T(3, 3)) {
}

bool Similarity3::equals(const Similarity3& other, double tol) const {
  return R_.equals(other.R_, tol) && traits<Point3>::Equals(t_, other.t_, tol)
      && s_ < (other.s_ + tol) && s_ > (other.s_ - tol);
}

bool Similarity3::operator==(const Similarity3& other) const {
  return R_.matrix() == other.R_.matrix() && t_ == other.t_ && s_ == other.s_;
}

void Similarity3::print(const std::string& s) const {
  std::cout << std::endl;
  std::cout << s;
  rotation().print("\nR:\n");
  std::cout << "t: " << translation().transpose() << " s: " << scale() << std::endl;
}

Similarity3 Similarity3::Identity() {
  return Similarity3();
}
Similarity3 Similarity3::operator*(const Similarity3& S) const {
  return Similarity3(R_ * S.R_, ((1.0 / S.s_) * t_) + R_ * S.t_, s_ * S.s_);
}

Similarity3 Similarity3::inverse() const {
  const Rot3 Rt = R_.inverse();
  const Point3 sRt = Rt * (-s_ * t_);
  return Similarity3(Rt, sRt, 1.0 / s_);
}

Point3 Similarity3::transformFrom(const Point3& p, //
    OptionalJacobian<3, 7> H1, OptionalJacobian<3, 3> H2) const {
  const Point3 q = R_ * p + t_;
  if (H1) {
    // For this derivative, see LieGroups.pdf
    const Matrix3 sR = s_ * R_.matrix();
    const Matrix3 DR = sR * skewSymmetric(-p.x(), -p.y(), -p.z());
    *H1 << DR, sR, sR * p;
  }
  if (H2)
    *H2 = s_ * R_.matrix(); // just 3*3 sub-block of matrix()
  return s_ * q;
}

Pose3 Similarity3::transformFrom(const Pose3& bTi, 
  OptionalJacobian<6, 7> Hself, OptionalJacobian<6, 6> H_bTi) const {

  // Similarity3 components and their derivatives with respect to the Similarity3.
  Matrix37 DSimR_dSim, Dt_dSim;
  Matrix17 Ds_dSim;
  const double Sim_s = this->scale(Hself ? &Ds_dSim : nullptr);
  const Point3 Sim_t = this->translation(Hself ? &Dt_dSim : nullptr);
  const Rot3 Sim_R = this->rotation(Hself ? &DSimR_dSim : nullptr);

  // Pose3 components and their derivatives with respect to the Pose3.
  Matrix36 Dp_dT, DTR_dT;
  const Rot3 TR = bTi.rotation(H_bTi ? &DTR_dT : nullptr);
  const Point3 p = bTi.translation(H_bTi ? &Dp_dT : nullptr);

  // Rotation component of the result.
  Matrix3 DR_dSimR, DR_dTR;
  const Rot3 R = Sim_R.compose(TR, Hself ? &DR_dSimR : nullptr, H_bTi ? &DR_dTR : nullptr);

  Matrix3 Dat_dSimR, Dat_dp;
  const Point3 at = Sim_R.rotate(p, Hself ? &Dat_dSimR : nullptr, H_bTi ? &Dat_dp: nullptr);
  const Point3 t = Sim_s * (at + Sim_t);

  if (Hself) {
    Hself->setZero();
    // Rotation component - this is correct.
    Hself->block<3, 7>(0, 0) = DR_dSimR * DSimR_dSim;

    // Translation component - this is not correct, needs to be fixed.
    Hself->block<3, 3>(3, 0) = Sim_s * (Dat_dSimR * DSimR_dSim).block<3, 3>(0, 0);
    
    // Part 1
    const auto ds = (at + Sim_t) * Ds_dSim; // 3x7
    // part 2
    const auto dt = Sim_s * Dt_dSim; // 3x7
    Hself->block<3, 4>(3, 3) = (ds + dt).block<3, 4>(0, 3);
  }

  if (H_bTi) {
    H_bTi->setZero();
    // Rotation component - this is correct.
    H_bTi->block<3, 6>(0, 0) = DR_dTR * DTR_dT;

    // H_bTi->block<3, 6>(3, 0) = Sim_s * Dat_dp * Dp_dT; // This does not work, dont know why.
    H_bTi->block<3, 3>(3, 3) = Sim_s * I_3x3; // This works, dont know why.

  }
  return Pose3(R, t);
}

Point3 Similarity3::operator*(const Point3& p) const {
  return transformFrom(p);
}

Similarity3 Similarity3::Align(const Point3Pairs &abPointPairs) {
  // Refer to Chapter 3 of
  // http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2005/Zinsser05-PSR.pdf
  if (abPointPairs.size() < 3)
    throw std::runtime_error("input should have at least 3 pairs of points");
  auto centroids = means(abPointPairs);
  auto d_abPointPairs = internal::subtractCentroids(abPointPairs, centroids);
  Matrix3 H = internal::calculateH(d_abPointPairs);
  // ClosestTo finds rotation matrix closest to H in Frobenius sense
  Rot3 aRb = Rot3::ClosestTo(H);
  return internal::align(d_abPointPairs, aRb, centroids);
}

Similarity3 Similarity3::Align(const Pose3Pairs &abPosePairs) {
  const size_t n = abPosePairs.size();
  if (n < 2)
    throw std::runtime_error("input should have at least 2 pairs of poses");

  // calculate rotation
  std::vector<Rot3> rotations;
  Point3Pairs abPointPairs;
  rotations.reserve(n);
  abPointPairs.reserve(n);
  // Below denotes the pose of the i'th object/camera/etc in frame "a" or frame "b"
  Pose3 aTi, bTi;
  for (const auto &[aTi, bTi] : abPosePairs) {
    const Rot3 aRb = aTi.rotation().compose(bTi.rotation().inverse());
    rotations.emplace_back(aRb);
    abPointPairs.emplace_back(aTi.translation(), bTi.translation());
  }
  const Rot3 aRb_estimate = FindKarcherMean<Rot3>(rotations);

  return internal::alignGivenR(abPointPairs, aRb_estimate);
}

Matrix4 Similarity3::Hat(const Vector7 &xi) {
  // http://www.ethaneade.org/latex2html/lie/node29.html
  const auto w = xi.head<3>();
  const auto u = xi.segment<3>(3);
  const double lambda = xi[6];
  Matrix4 W;
  W << skewSymmetric(w), u, 0, 0, 0, -lambda;
  return W;
}

Vector7 Similarity3::Vee(const Matrix4 &Xi) {
  Vector7 xi;
  xi.head<3>() = Rot3::Vee(Xi.topLeftCorner<3, 3>());
  xi.segment<3>(3) = Xi.topRightCorner<3, 1>();
  xi[6] = -Xi(3, 3);
  return xi;
}

Matrix7 Similarity3::AdjointMap() const {
  // http://www.ethaneade.org/latex2html/lie/node30.html
  const Matrix3 R = R_.matrix();
  const Vector3 t = t_;
  const Matrix3 A = s_ * skewSymmetric(t) * R;
  Matrix7 adj;
  adj << R, Z_3x3, Matrix31::Zero(), // 3*7
  A, s_ * R, -s_ * t, // 3*7
  Matrix16::Zero(), 1; // 1*7
  return adj;
}

namespace {
// Functor that implements the Similarity3 V(ω, λ) kernel:
// See http://www.ethaneade.org/latex2html/lie/node29.html
struct LocalV : public so3::DexpFunctor {
  double lambda;  ///< scale log parameter
  double alpha{0}, beta{0}, mu{0};
  double P{0}, Q{0}, R{0};

  explicit LocalV(const Vector3& omega, double lambda,
                  double nearZeroThresholdSq, double nearPiThresholdSq)
      : so3::DexpFunctor(omega, nearZeroThresholdSq, nearPiThresholdSq),
        lambda(lambda) {
    compute_();
  }

  explicit LocalV(const Vector3& omega, double lambda)
      : so3::DexpFunctor(omega), lambda(lambda) {
    compute_();
  }

  void compute_() {
    const double lambda2 = lambda * lambda, lambda3 = lambda2 * lambda;
    if (lambda2 > 1e-9) {
      const double e = std::exp(-lambda);
      P = (1.0 - e) / lambda;
      alpha = 1.0 / (1.0 + (theta2 / lambda2));  // = λ²/(λ²+θ²)
      beta = (e - 1.0 + lambda) / lambda2;
      mu = (1.0 - lambda + 0.5 * lambda2 - e) / lambda3;
    } else {
      P = 1.0 - lambda / 2.0 + lambda2 / 6.0;
      alpha = 0.0;
      beta = 0.5 - lambda / 6.0 + lambda2 / 24.0 - lambda3 / 120.0;
      mu = 1.0 / 6.0 - lambda / 24.0 + lambda2 / 120.0 - lambda3 / 720.0;
    }
    const double one_minus_alpha = 1.0 - alpha;
    Q = alpha * beta + one_minus_alpha * (B - lambda * C());
    R = alpha * mu + one_minus_alpha * (C() - lambda * E());
  }

  Matrix3 V() const { return P * I_3x3 + Q * W + R * WW; }

  so3::Kernel kernel() const {
    const double lambda2 = lambda * lambda;
    const double dalpha =
        (lambda2 > 1e-9) ? (-2.0 * alpha * alpha / lambda2) : 0.0;
    const double dQ = (beta - (B - lambda * C())) * dalpha +
                      (1.0 - alpha) * (dB() - lambda * dC());
    const double dR = (mu - (C() - lambda * E())) * dalpha +
                      (1.0 - alpha) * (dC() - lambda * dE());
    return so3::Kernel{this, P, Q, R, dQ, dR};
  }
};
}  // namespace

Matrix3 Similarity3::GetV(Vector3 w, double lambda) {
  return LocalV(w, lambda).V();
}
Vector7 Similarity3::Logmap(const Similarity3& T, OptionalJacobian<7, 7> Hm) {
  // To get the logmap, calculate w and lambda, then solve for u as shown by Ethan at
  // www.ethaneade.org/latex2html/lie/node29.html
  const Vector3 w = Rot3::Logmap(T.R_);
  const double lambda = log(T.s_);
  Vector7 result;
  result << w, GetV(w, lambda).inverse() * T.t_, lambda;
  if (Hm) {
    throw std::runtime_error("Similarity3::Logmap: derivative not implemented");
  }
  return result;
}

Similarity3 Similarity3::Expmap(const Vector7& v, OptionalJacobian<7, 7> Hm) {
  const auto w = v.head<3>();
  const auto rho = v.segment<3>(3);
  const double lambda = v[6];
  if (Hm) {
    throw std::runtime_error("Similarity3::Expmap: derivative not implemented");
  }
  const Matrix3 V = GetV(w, lambda);
  return Similarity3(Rot3::Expmap(w), Point3(V * rho), exp(lambda));
}

std::ostream &operator<<(std::ostream &os, const Similarity3& p) {
  os << "[" << p.rotation().xyz().transpose() << " "
      << p.translation().transpose() << " " << p.scale() << "]\';";
  return os;
}

Matrix4 Similarity3::matrix() const {
  Matrix4 T;
  T.topRows<3>() << R_.matrix(), t_;
  T.bottomRows<1>() << 0, 0, 0, 1.0 / s_;
  return T;
}




} // namespace gtsam
