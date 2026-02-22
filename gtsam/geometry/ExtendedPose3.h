/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    ExtendedPose3.h
 * @brief   Extended pose Lie group SE_k(3), with static or dynamic k.
 * @author  Frank Dellaert, et al.
 */

#pragma once

#include <gtsam/base/MatrixLieGroup.h>
#include <gtsam/geometry/Kernel.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Rot3.h>

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#if GTSAM_ENABLE_BOOST_SERIALIZATION
#include <boost/serialization/nvp.hpp>
#endif

namespace gtsam {

namespace internal {

/// Manifold dimensionality for SE_k(3): 3 + 3*k.
constexpr int DimensionExtendedPose3(int k) {
  return (k == Eigen::Dynamic) ? Eigen::Dynamic : 3 + 3 * k;
}

/// Homogeneous matrix size for SE_k(3): (3 + k) x (3 + k).
constexpr int MatrixDimExtendedPose3(int k) {
  return (k == Eigen::Dynamic) ? Eigen::Dynamic : 3 + k;
}

}  // namespace internal

/**
 * Lie group SE_k(3): semidirect product of SO(3) with k copies of R^3.
 * Tangent ordering is [omega, x1, x2, ..., xk], each xi in R^3.
 *
 * Template parameter K can be fixed (K >= 1) or Eigen::Dynamic.
 */
template <int K>
class GTSAM_EXPORT ExtendedPose3
    : public MatrixLieGroup<ExtendedPose3<K>, internal::DimensionExtendedPose3(K),
                            internal::MatrixDimExtendedPose3(K)> {
 public:
  inline constexpr static auto dimension = internal::DimensionExtendedPose3(K);
  inline constexpr static auto matrix_dim = internal::MatrixDimExtendedPose3(K);

  using Base = MatrixLieGroup<ExtendedPose3<K>, dimension, matrix_dim>;
  using TangentVector = typename Base::TangentVector;
  using Jacobian = typename Base::Jacobian;
  using ChartJacobian = typename Base::ChartJacobian;
  using ComponentJacobian =
      std::conditional_t<dimension == Eigen::Dynamic,
                         OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic>,
                         OptionalJacobian<3, dimension>>;
  using LieAlgebra = Eigen::Matrix<double, matrix_dim, matrix_dim>;
  using Matrix3K = Eigen::Matrix<double, 3, K>;

  static_assert(K == Eigen::Dynamic || K >= 1,
                "ExtendedPose3<K>: K should be >= 1 or Eigen::Dynamic.");

 protected:
  Rot3 R_;      ///< Rotation component.
  Matrix3K x_;  ///< K translation-like columns in world frame.

  template <int K_>
  using IsDynamic = typename std::enable_if<K_ == Eigen::Dynamic, void>::type;
  template <int K_>
  using IsFixed = typename std::enable_if<K_ >= 1, void>::type;

  static size_t RuntimeK(const TangentVector& xi) {
    if constexpr (K == Eigen::Dynamic) {
      if (xi.size() < 3 || (xi.size() - 3) % 3 != 0) {
        throw std::invalid_argument(
            "ExtendedPose3: tangent vector size must be 3 + 3*k.");
      }
      return static_cast<size_t>((xi.size() - 3) / 3);
    } else {
      return static_cast<size_t>(K);
    }
  }

  static void ZeroJacobian(ChartJacobian H, size_t d) {
    if (!H) return;
    if constexpr (dimension == Eigen::Dynamic) {
      H->setZero(d, d);
    } else {
      (void)d;
      H->setZero();
    }
  }

 public:
  /// @name Constructors
  /// @{

  /// Construct fixed-size identity.
  template <int K_ = K, typename = IsFixed<K_>>
  ExtendedPose3() : R_(Rot3::Identity()), x_(Matrix3K::Zero()) {}

  /// Construct dynamic identity, optionally with runtime k.
  template <int K_ = K, typename = IsDynamic<K_>>
  explicit ExtendedPose3(size_t k = 0) : R_(Rot3::Identity()), x_(3, k) {
    x_.setZero();
  }

  ExtendedPose3(const ExtendedPose3&) = default;
  ExtendedPose3& operator=(const ExtendedPose3&) = default;

  /// Construct from rotation and 3xK block.
  ExtendedPose3(const Rot3& R, const Matrix3K& x) : R_(R), x_(x) {}

  /// Construct from homogeneous matrix representation.
  explicit ExtendedPose3(const LieAlgebra& T) {
    if constexpr (K == Eigen::Dynamic) {
      const auto n = T.rows();
      if (T.cols() != n || n < 3) {
        throw std::invalid_argument("ExtendedPose3: invalid matrix shape.");
      }
      const auto k = n - 3;
      x_.resize(3, k);
      x_.setZero();
    }

    const auto n = T.rows();
    if constexpr (K != Eigen::Dynamic) {
      if (n != matrix_dim || T.cols() != matrix_dim) {
        throw std::invalid_argument("ExtendedPose3: invalid matrix shape.");
      }
    } else {
      if (T.cols() != n || n < 3) {
        throw std::invalid_argument("ExtendedPose3: invalid matrix shape.");
      }
    }

    R_ = Rot3(T.template block<3, 3>(0, 0));
    x_ = T.block(0, 3, 3, n - 3);
  }

  /// @}
  /// @name Access
  /// @{

  static size_t Dimension(size_t k) { return 3 + 3 * k; }

  /// Number of R^3 columns.
  size_t k() const { return static_cast<size_t>(x_.cols()); }

  /// Runtime manifold dimension.
  template <int K_ = K, typename = IsDynamic<K_>>
  size_t dim() const {
    return Dimension(k());
  }

  /// Rotation component.
  const Rot3& rotation(ComponentJacobian H = {}) const {
    if (H) {
      if constexpr (dimension == Eigen::Dynamic) {
        H->setZero(3, dim());
      } else {
        H->setZero();
      }
      H->block(0, 0, 3, 3) = I_3x3;
    }
    return R_;
  }

  /// i-th R^3 component, returned by value.
  Point3 x(size_t i, ComponentJacobian H = {}) const {
    if (i >= k()) throw std::out_of_range("ExtendedPose3: x(i) out of range.");
    if (H) {
      if constexpr (dimension == Eigen::Dynamic) {
        H->setZero(3, dim());
      } else {
        H->setZero();
      }
      H->block(0, 3 + 3 * i, 3, 3) = R_.matrix();
    }
    return x_.col(static_cast<Eigen::Index>(i));
  }

  const Matrix3K& xMatrix() const { return x_; }
  Matrix3K& xMatrix() { return x_; }

  /// @}
  /// @name Testable
  /// @{

  void print(const std::string& s = "") const {
    std::cout << (s.empty() ? s : s + " ") << *this << std::endl;
  }

  bool equals(const ExtendedPose3& other, double tol = 1e-9) const {
    return R_.equals(other.R_, tol) && equal_with_abs_tol(x_, other.x_, tol);
  }

  /// @}
  /// @name Group
  /// @{

  template <int K_ = K, typename = IsFixed<K_>>
  static ExtendedPose3 Identity() {
    return ExtendedPose3();
  }

  template <int K_ = K, typename = IsDynamic<K_>>
  static ExtendedPose3 Identity(size_t k = 0) {
    return ExtendedPose3(k);
  }

  ExtendedPose3 inverse() const {
    const Rot3 Rt = R_.inverse();
    const Matrix3K x = -(Rt.matrix() * x_);
    return ExtendedPose3(Rt, x);
  }

  ExtendedPose3 operator*(const ExtendedPose3& other) const {
    if (k() != other.k()) {
      throw std::invalid_argument(
          "ExtendedPose3: compose requires matching k.");
    }
    Matrix3K x = x_ + R_.matrix() * other.x_;
    return ExtendedPose3(R_ * other.R_, x);
  }

  /// @}
  /// @name Lie Group
  /// @{

  static ExtendedPose3 Expmap(const TangentVector& xi, ChartJacobian Hxi = {}) {
    const size_t k = RuntimeK(xi);

    const Vector3 w = xi.template head<3>();
    const so3::DexpFunctor local(w);
    const Rot3 R(local.expmap());
    const Matrix3 Rt = R.transpose();

    Matrix3K x;
    if constexpr (K == Eigen::Dynamic) x.resize(3, static_cast<Eigen::Index>(k));

    if (Hxi) {
      const size_t d = 3 + 3 * k;
      ZeroJacobian(Hxi, d);
      const Matrix3 Jr = local.Jacobian().right();
      Hxi->block(0, 0, 3, 3) = Jr;
      for (size_t i = 0; i < k; ++i) {
        Matrix3 H_xi_w;
        const Vector3 rho = xi.template segment<3>(3 + 3 * i);
        x.col(static_cast<Eigen::Index>(i)) =
            local.Jacobian().applyLeft(rho, &H_xi_w);
        Hxi->block(3 + 3 * i, 0, 3, 3) = Rt * H_xi_w;
        Hxi->block(3 + 3 * i, 3 + 3 * i, 3, 3) = Jr;
      }
    } else {
      for (size_t i = 0; i < k; ++i) {
        const Vector3 rho = xi.template segment<3>(3 + 3 * i);
        x.col(static_cast<Eigen::Index>(i)) = local.Jacobian().applyLeft(rho);
      }
    }

    return ExtendedPose3(R, x);
  }

  static TangentVector Logmap(const ExtendedPose3& pose, ChartJacobian Hpose = {}) {
    const Vector3 w = Rot3::Logmap(pose.R_);
    const so3::DexpFunctor local(w);

    TangentVector xi;
    if constexpr (K == Eigen::Dynamic) xi.resize(static_cast<Eigen::Index>(pose.dim()));
    xi.template head<3>() = w;
    for (size_t i = 0; i < pose.k(); ++i) {
      xi.template segment<3>(3 + 3 * i) =
          local.InvJacobian().applyLeft(pose.x_.col(static_cast<Eigen::Index>(i)));
    }

    if (Hpose) *Hpose = LogmapDerivative(xi);
    return xi;
  }

  Jacobian AdjointMap() const {
    const Matrix3 R = R_.matrix();
    Jacobian adj;
    if constexpr (dimension == Eigen::Dynamic) {
      adj.setZero(dim(), dim());
    } else {
      adj.setZero();
    }

    adj.block(0, 0, 3, 3) = R;
    for (size_t i = 0; i < k(); ++i) {
      adj.block(3 + 3 * i, 0, 3, 3) =
          skewSymmetric(x_.col(static_cast<Eigen::Index>(i))) * R;
      adj.block(3 + 3 * i, 3 + 3 * i, 3, 3) = R;
    }
    return adj;
  }

  TangentVector Adjoint(const TangentVector& xi_b,
                        ChartJacobian H_this = {},
                        ChartJacobian H_xib = {}) const {
    const Jacobian Ad = AdjointMap();
    if (H_this) *H_this = -Ad * adjointMap(xi_b);
    if (H_xib) *H_xib = Ad;
    return Ad * xi_b;
  }

  static Jacobian adjointMap(const TangentVector& xi) {
    const size_t k = RuntimeK(xi);
    const Matrix3 w_hat = skewSymmetric(xi(0), xi(1), xi(2));

    Jacobian adj;
    if constexpr (dimension == Eigen::Dynamic) {
      adj.setZero(3 + 3 * k, 3 + 3 * k);
    } else {
      adj.setZero();
    }

    adj.block(0, 0, 3, 3) = w_hat;
    for (size_t i = 0; i < k; ++i) {
      adj.block(3 + 3 * i, 0, 3, 3) =
          skewSymmetric(xi(3 + 3 * i + 0), xi(3 + 3 * i + 1), xi(3 + 3 * i + 2));
      adj.block(3 + 3 * i, 3 + 3 * i, 3, 3) = w_hat;
    }
    return adj;
  }

  static TangentVector adjoint(const TangentVector& xi, const TangentVector& y,
                               ChartJacobian Hxi = {}, ChartJacobian H_y = {}) {
    const Jacobian ad_xi = adjointMap(xi);
    if (Hxi) {
      if constexpr (dimension == Eigen::Dynamic) {
        Hxi->setZero(xi.size(), xi.size());
      } else {
        Hxi->setZero();
      }
      for (Eigen::Index i = 0; i < xi.size(); ++i) {
        TangentVector dxi;
        if constexpr (dimension == Eigen::Dynamic) {
          dxi = TangentVector::Zero(xi.size());
        } else {
          dxi = TangentVector::Zero();
        }
        dxi(i) = 1.0;
        Hxi->col(i) = adjointMap(dxi) * y;
      }
    }
    if (H_y) *H_y = ad_xi;
    return ad_xi * y;
  }

  static Jacobian ExpmapDerivative(const TangentVector& xi) {
    Jacobian J;
    Expmap(xi, J);
    return J;
  }

  static Jacobian LogmapDerivative(const TangentVector& xi) {
    const size_t k = RuntimeK(xi);
    const Vector3 w = xi.template head<3>();
    const so3::DexpFunctor local(w);
    const Matrix3 Rt = local.expmap().transpose();
    const Matrix3 Jw = Rot3::LogmapDerivative(w);

    Jacobian J;
    if constexpr (dimension == Eigen::Dynamic) {
      J.setZero(3 + 3 * k, 3 + 3 * k);
    } else {
      J.setZero();
    }

    J.block(0, 0, 3, 3) = Jw;
    for (size_t i = 0; i < k; ++i) {
      Matrix3 H_xi_w;
      local.Jacobian().applyLeft(xi.template segment<3>(3 + 3 * i), H_xi_w);
      const Matrix3 Q = Rt * H_xi_w;
      J.block(3 + 3 * i, 0, 3, 3) = -Jw * Q * Jw;
      J.block(3 + 3 * i, 3 + 3 * i, 3, 3) = Jw;
    }
    return J;
  }

  static Jacobian LogmapDerivative(const ExtendedPose3& pose) {
    return LogmapDerivative(Logmap(pose));
  }

  struct ChartAtOrigin {
    static ExtendedPose3 Retract(const TangentVector& xi, ChartJacobian Hxi = {}) {
      return Expmap(xi, Hxi);
    }

    static TangentVector Local(const ExtendedPose3& pose, ChartJacobian Hpose = {}) {
      return Logmap(pose, Hpose);
    }
  };

  using LieGroup<ExtendedPose3<K>, dimension>::inverse;

  /// @}
  /// @name Matrix Lie Group
  /// @{

  LieAlgebra matrix() const {
    LieAlgebra M;
    if constexpr (matrix_dim == Eigen::Dynamic) {
      const Eigen::Index n = 3 + static_cast<Eigen::Index>(k());
      M = LieAlgebra::Identity(n, n);
    } else {
      M = LieAlgebra::Identity();
    }
    M.template block<3, 3>(0, 0) = R_.matrix();
    M.block(0, 3, 3, static_cast<Eigen::Index>(k())) = x_;
    return M;
  }

  static LieAlgebra Hat(const TangentVector& xi) {
    const size_t k = RuntimeK(xi);
    LieAlgebra X;
    if constexpr (matrix_dim == Eigen::Dynamic) {
      X.setZero(3 + k, 3 + k);
    } else {
      X.setZero();
    }
    X.block(0, 0, 3, 3) = skewSymmetric(xi(0), xi(1), xi(2));
    for (size_t i = 0; i < k; ++i) {
      X.block(0, 3 + i, 3, 1) = xi.template segment<3>(3 + 3 * i);
    }
    return X;
  }

  static TangentVector Vee(const LieAlgebra& X) {
    const Eigen::Index k = X.cols() - 3;
    TangentVector xi;
    if constexpr (dimension == Eigen::Dynamic) {
      xi.resize(3 + 3 * k);
    }
    xi(0) = X(2, 1);
    xi(1) = X(0, 2);
    xi(2) = X(1, 0);
    for (Eigen::Index i = 0; i < k; ++i) {
      xi.template segment<3>(3 + 3 * i) = X.template block<3, 1>(0, 3 + i);
    }
    return xi;
  }

  /// @}

  friend std::ostream& operator<<(std::ostream& os, const ExtendedPose3& p) {
    os << "R: " << p.R_ << "\n";
    os << "x: " << p.x_;
    return os;
  }

 private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int /*version*/) {
    ar& BOOST_SERIALIZATION_NVP(R_);
    ar& BOOST_SERIALIZATION_NVP(x_);
  }
#endif
};

/// Convenience typedef for dynamic k.
using ExtendedPose3Dynamic = ExtendedPose3<Eigen::Dynamic>;

template <int K>
struct traits<ExtendedPose3<K>>
    : public internal::MatrixLieGroup<ExtendedPose3<K>,
                                      internal::MatrixDimExtendedPose3(K)> {};

template <int K>
struct traits<const ExtendedPose3<K>>
    : public internal::MatrixLieGroup<ExtendedPose3<K>,
                                      internal::MatrixDimExtendedPose3(K)> {};

}  // namespace gtsam
