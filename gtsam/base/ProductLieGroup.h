/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file ProductLieGroup.h
 * @date May, 2015
 * @author Frank Dellaert
 * @brief Group product of two Lie Groups
 */

#pragma once

#include <gtsam/base/Lie.h>
#include <gtsam/base/Testable.h>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>  // pair

namespace gtsam {

/**
 * @brief Template to construct the product Lie group of two other Lie groups
 * Assumes Lie group structure for G and H
 */
template <typename G, typename H>
class ProductLieGroup : public std::pair<G, H> {
  GTSAM_CONCEPT_ASSERT(IsLieGroup<G>);
  GTSAM_CONCEPT_ASSERT(IsLieGroup<H>);
  GTSAM_CONCEPT_ASSERT(IsTestable<G>);
  GTSAM_CONCEPT_ASSERT(IsTestable<H>);

 public:
  /// Base pair type
  typedef std::pair<G, H> Base;

 protected:
  /// Dimensions of the two subgroups
  inline constexpr static int dimension1 = traits<G>::dimension;
  inline constexpr static int dimension2 = traits<H>::dimension;
  inline constexpr static bool firstDynamic = dimension1 == Eigen::Dynamic;
  inline constexpr static bool secondDynamic = dimension2 == Eigen::Dynamic;

 public:
  /// Manifold dimension
  inline constexpr static int dimension =
      firstDynamic || secondDynamic ? Eigen::Dynamic : dimension1 + dimension2;

  /// Tangent vector type
  using TangentVector = std::conditional_t<dimension == Eigen::Dynamic, Vector,
                                           Eigen::Matrix<double, dimension, 1>>;

  /// Chart Jacobian type
  using ChartJacobian =
      std::conditional_t<dimension == Eigen::Dynamic,
                         OptionalJacobian<Eigen::Dynamic, Eigen::Dynamic>,
                         OptionalJacobian<dimension, dimension>>;

  /// Jacobian types for internal use
  using Jacobian =
      std::conditional_t<dimension == Eigen::Dynamic, Matrix,
                         Eigen::Matrix<double, dimension, dimension>>;
  using Jacobian1 = typename traits<G>::Jacobian;
  using Jacobian2 = typename traits<H>::Jacobian;

 public:
  /// @name Standard Constructors
  /// @{

  /// Default constructor yields identity
  ProductLieGroup() : Base(defaultIdentity<G>(), defaultIdentity<H>()) {}

  /// Construct from two subgroup elements
  ProductLieGroup(const G& g, const H& h) : Base(g, h) {}

  /// Construct from base pair
  ProductLieGroup(const Base& base) : Base(base) {}

  /// @}
  /// @name Group Operations
  /// @{

  typedef multiplicative_group_tag group_flavor;

  /// Identity element
  static ProductLieGroup Identity() { return ProductLieGroup(); }

  /// Group multiplication
  ProductLieGroup operator*(const ProductLieGroup& other) const {
    checkMatchingDimensions(firstDim(), secondDim(), other.firstDim(),
                            other.secondDim(), "operator*");
    return ProductLieGroup(traits<G>::Compose(this->first, other.first),
                           traits<H>::Compose(this->second, other.second));
  }

  /// Group inverse
  ProductLieGroup inverse() const {
    return ProductLieGroup(traits<G>::Inverse(this->first),
                           traits<H>::Inverse(this->second));
  }

  /// Compose with another element (same as operator*)
  ProductLieGroup compose(const ProductLieGroup& g) const {
    return (*this) * g;
  }

  /// Calculate relative transformation
  ProductLieGroup between(const ProductLieGroup& g) const {
    return this->inverse() * g;
  }

  /// @}
  /// @name Manifold Operations
  /// @{

  /// Manifold dimension
  inline constexpr static int manifoldDimension = dimension;

  /// Return manifold dimension
  static constexpr int Dim() { return manifoldDimension; }

  /// Return manifold dimension
  size_t dim() const { return combinedDimension(firstDim(), secondDim()); }

  /// Retract to manifold
  ProductLieGroup retract(const TangentVector& v, ChartJacobian H1 = {},
                          ChartJacobian H2 = {}) const {
    if (H1 || H2) {
      throw std::runtime_error(
          "ProductLieGroup::retract derivatives not implemented yet");
    }
    const size_t firstDimension = firstDim();
    const size_t secondDimension = secondDim();
    if (static_cast<size_t>(v.size()) !=
        combinedDimension(firstDimension, secondDimension)) {
      throw std::invalid_argument(
          "ProductLieGroup::retract tangent dimension does not match product "
          "dimension");
    }
    G g = traits<G>::Retract(this->first,
                             tangentSegment<G>(v, 0, firstDimension));
    H h = traits<H>::Retract(
        this->second, tangentSegment<H>(v, firstDimension, secondDimension));
    return ProductLieGroup(g, h);
  }

  /// Local coordinates on manifold
  TangentVector localCoordinates(const ProductLieGroup& g,
                                 ChartJacobian H1 = {},
                                 ChartJacobian H2 = {}) const {
    if (H1 || H2) {
      throw std::runtime_error(
          "ProductLieGroup::localCoordinates derivatives not implemented yet");
    }
    checkMatchingDimensions(firstDim(), secondDim(), g.firstDim(),
                            g.secondDim(), "localCoordinates");
    const size_t firstDimension = firstDim();
    const size_t secondDimension = secondDim();
    typename traits<G>::TangentVector v1 =
        traits<G>::Local(this->first, g.first);
    typename traits<H>::TangentVector v2 =
        traits<H>::Local(this->second, g.second);
    return makeTangentVector(v1, v2, firstDimension, secondDimension);
  }

  /// @}
  /// @name Lie Group Operations
  /// @{

  /// Compose with Jacobians
  ProductLieGroup compose(const ProductLieGroup& other, ChartJacobian H1,
                          ChartJacobian H2 = {}) const {
    checkMatchingDimensions(firstDim(), secondDim(), other.firstDim(),
                            other.secondDim(), "compose");
    const size_t firstDimension = firstDim();
    const size_t secondDimension = secondDim();
    const size_t productDimension =
        combinedDimension(firstDimension, secondDimension);
    Jacobian1 D_g_first;
    Jacobian2 D_h_second;
    G g =
        traits<G>::Compose(this->first, other.first, H1 ? &D_g_first : nullptr);
    H h = traits<H>::Compose(this->second, other.second,
                             H1 ? &D_h_second : nullptr);
    if (H1) {
      *H1 = zeroJacobian(productDimension);
      H1->block(0, 0, firstDimension, firstDimension) = D_g_first;
      H1->block(firstDimension, firstDimension, secondDimension,
                secondDimension) = D_h_second;
    }
    if (H2) *H2 = identityJacobian(productDimension);
    return ProductLieGroup(g, h);
  }

  /// Between with Jacobians
  ProductLieGroup between(const ProductLieGroup& other, ChartJacobian H1,
                          ChartJacobian H2 = {}) const {
    checkMatchingDimensions(firstDim(), secondDim(), other.firstDim(),
                            other.secondDim(), "between");
    const size_t firstDimension = firstDim();
    const size_t secondDimension = secondDim();
    const size_t productDimension =
        combinedDimension(firstDimension, secondDimension);
    Jacobian1 D_g_first;
    Jacobian2 D_h_second;
    G g =
        traits<G>::Between(this->first, other.first, H1 ? &D_g_first : nullptr);
    H h = traits<H>::Between(this->second, other.second,
                             H1 ? &D_h_second : nullptr);
    if (H1) {
      *H1 = zeroJacobian(productDimension);
      H1->block(0, 0, firstDimension, firstDimension) = D_g_first;
      H1->block(firstDimension, firstDimension, secondDimension,
                secondDimension) = D_h_second;
    }
    if (H2) *H2 = identityJacobian(productDimension);
    return ProductLieGroup(g, h);
  }

  /// Inverse with Jacobian
  ProductLieGroup inverse(ChartJacobian D) const {
    const size_t firstDimension = firstDim();
    const size_t secondDimension = secondDim();
    const size_t productDimension =
        combinedDimension(firstDimension, secondDimension);
    Jacobian1 D_g_first;
    Jacobian2 D_h_second;
    G g = traits<G>::Inverse(this->first, D ? &D_g_first : nullptr);
    H h = traits<H>::Inverse(this->second, D ? &D_h_second : nullptr);
    if (D) {
      *D = zeroJacobian(productDimension);
      D->block(0, 0, firstDimension, firstDimension) = D_g_first;
      D->block(firstDimension, firstDimension, secondDimension,
               secondDimension) = D_h_second;
    }
    return ProductLieGroup(g, h);
  }

  /// Exponential map
  static ProductLieGroup Expmap(const TangentVector& v, ChartJacobian Hv = {}) {
    if constexpr (firstDynamic && secondDynamic) {
      if (v.size() == 0) {
        if (Hv) *Hv = Matrix::Zero(0, 0);
        return ProductLieGroup();
      }
      throw std::invalid_argument(
          "ProductLieGroup::Expmap requires split tangent vectors when both "
          "factors are dynamic");
    } else if constexpr (firstDynamic) {
      if (v.size() < dimension2) {
        throw std::invalid_argument(
            "ProductLieGroup::Expmap tangent dimension is too small for the "
            "fixed second factor");
      }
      const size_t firstDimension = static_cast<size_t>(v.size() - dimension2);
      return expmapWithDimensions(v, firstDimension,
                                  static_cast<size_t>(dimension2), Hv);
    } else if constexpr (secondDynamic) {
      if (v.size() < dimension1) {
        throw std::invalid_argument(
            "ProductLieGroup::Expmap tangent dimension is too small for the "
            "fixed first factor");
      }
      const size_t secondDimension = static_cast<size_t>(v.size() - dimension1);
      return expmapWithDimensions(v, static_cast<size_t>(dimension1),
                                  secondDimension, Hv);
    } else {
      return expmapWithDimensions(v, static_cast<size_t>(dimension1),
                                  static_cast<size_t>(dimension2), Hv);
    }
  }

  /// Exponential map from subgroup tangent vectors
  static ProductLieGroup Expmap(
      const Eigen::Ref<const typename traits<G>::TangentVector>& v1,
      const Eigen::Ref<const typename traits<H>::TangentVector>& v2,
      ChartJacobian Hv) {
    const size_t firstDimension = static_cast<size_t>(v1.size());
    const size_t secondDimension = static_cast<size_t>(v2.size());
    Jacobian1 D_g_first;
    Jacobian2 D_h_second;
    G g = traits<G>::Expmap(v1, Hv ? &D_g_first : nullptr);
    H h = traits<H>::Expmap(v2, Hv ? &D_h_second : nullptr);
    if (Hv) {
      *Hv = zeroJacobian(combinedDimension(firstDimension, secondDimension));
      Hv->block(0, 0, firstDimension, firstDimension) = D_g_first;
      Hv->block(firstDimension, firstDimension, secondDimension,
                secondDimension) = D_h_second;
    }
    return ProductLieGroup(g, h);
  }

  /// Exponential map from subgroup tangent vectors
  static ProductLieGroup Expmap(
      const Eigen::Ref<const typename traits<G>::TangentVector>& v1,
      const Eigen::Ref<const typename traits<H>::TangentVector>& v2) {
    return Expmap(v1, v2, {});
  }

  /// Logarithmic map
  static TangentVector Logmap(const ProductLieGroup& p, ChartJacobian Hp = {}) {
    const size_t firstDimension = p.firstDim();
    const size_t secondDimension = p.secondDim();
    const size_t productDimension =
        combinedDimension(firstDimension, secondDimension);
    Jacobian1 D_g_first;
    Jacobian2 D_h_second;
    typename traits<G>::TangentVector v1 =
        traits<G>::Logmap(p.first, Hp ? &D_g_first : nullptr);
    typename traits<H>::TangentVector v2 =
        traits<H>::Logmap(p.second, Hp ? &D_h_second : nullptr);
    TangentVector v =
        makeTangentVector(v1, v2, firstDimension, secondDimension);
    if (Hp) {
      *Hp = zeroJacobian(productDimension);
      Hp->block(0, 0, firstDimension, firstDimension) = D_g_first;
      Hp->block(firstDimension, firstDimension, secondDimension,
                secondDimension) = D_h_second;
    }
    return v;
  }

  /// Local coordinates (same as Logmap)
  static TangentVector LocalCoordinates(const ProductLieGroup& p,
                                        ChartJacobian Hp = {}) {
    return Logmap(p, Hp);
  }

  /// Right multiplication by exponential map
  ProductLieGroup expmap(const TangentVector& v) const {
    return compose(expmapWithDimensions(v, firstDim(), secondDim()));
  }

  /// Logarithmic map for relative transformation
  TangentVector logmap(const ProductLieGroup& g) const {
    return ProductLieGroup::Logmap(between(g));
  }

  /// Adjoint map
  Jacobian AdjointMap() const {
    const auto adjG = traits<G>::AdjointMap(this->first);
    const auto adjH = traits<H>::AdjointMap(this->second);
    const size_t d1 = static_cast<size_t>(adjG.rows());
    const size_t d2 = static_cast<size_t>(adjH.rows());
    Jacobian adj = zeroJacobian(d1 + d2);
    adj.block(0, 0, d1, d1) = adjG;
    adj.block(d1, d1, d2, d2) = adjH;
    return adj;
  }

  /// @}

 protected:
  template <typename T>
  static T defaultIdentity() {
    if constexpr (traits<T>::dimension == Eigen::Dynamic) {
      return T();
    } else {
      return traits<T>::Identity();
    }
  }

  size_t firstDim() const { return traits<G>::GetDimension(this->first); }
  size_t secondDim() const { return traits<H>::GetDimension(this->second); }

  static size_t combinedDimension(size_t d1, size_t d2) { return d1 + d2; }

  template <typename T, int Dim = traits<T>::dimension>
  static typename traits<T>::TangentVector tangentSegment(
      const TangentVector& v, size_t start, size_t runtimeDimension) {
    const int startIndex = static_cast<int>(start);
    const int runtimeIndex = static_cast<int>(runtimeDimension);
    if constexpr (Dim == Eigen::Dynamic) {
      return v.segment(startIndex, runtimeIndex);
    } else {
      static_cast<void>(runtimeDimension);
      return v.template segment<Dim>(startIndex);
    }
  }

  static TangentVector makeTangentVector(
      const typename traits<G>::TangentVector& v1,
      const typename traits<H>::TangentVector& v2, size_t firstDimension,
      size_t secondDimension) {
    const int firstIndex = static_cast<int>(firstDimension);
    const int secondIndex = static_cast<int>(secondDimension);
    if constexpr (dimension == Eigen::Dynamic) {
      TangentVector v(combinedDimension(firstDimension, secondDimension));
      v.segment(0, firstIndex) = v1;
      v.segment(firstIndex, secondIndex) = v2;
      return v;
    } else {
      static_cast<void>(firstDimension);
      static_cast<void>(secondDimension);
      TangentVector v;
      v << v1, v2;
      return v;
    }
  }

  static Jacobian zeroJacobian(size_t productDimension) {
    if constexpr (dimension == Eigen::Dynamic) {
      return Jacobian::Zero(productDimension, productDimension);
    } else {
      static_cast<void>(productDimension);
      return Jacobian::Zero();
    }
  }

  static Jacobian identityJacobian(size_t productDimension) {
    if constexpr (dimension == Eigen::Dynamic) {
      return Jacobian::Identity(productDimension, productDimension);
    } else {
      static_cast<void>(productDimension);
      return Jacobian::Identity();
    }
  }

  static void checkMatchingDimensions(size_t first1, size_t second1,
                                      size_t first2, size_t second2,
                                      const char* operation) {
    if (first1 != first2 || second1 != second2) {
      throw std::invalid_argument(std::string("ProductLieGroup::") + operation +
                                  " requires matching component dimensions");
    }
  }

  static ProductLieGroup expmapWithDimensions(const TangentVector& v,
                                              size_t firstDimension,
                                              size_t secondDimension,
                                              ChartJacobian Hv = {}) {
    if (static_cast<size_t>(v.size()) !=
        combinedDimension(firstDimension, secondDimension)) {
      throw std::invalid_argument(
          "ProductLieGroup::Expmap tangent dimension does not match requested "
          "component dimensions");
    }
    Jacobian1 D_g_first;
    Jacobian2 D_h_second;
    G g = traits<G>::Expmap(tangentSegment<G>(v, 0, firstDimension),
                            Hv ? &D_g_first : nullptr);
    H h =
        traits<H>::Expmap(tangentSegment<H>(v, firstDimension, secondDimension),
                          Hv ? &D_h_second : nullptr);
    if (Hv) {
      *Hv = zeroJacobian(combinedDimension(firstDimension, secondDimension));
      Hv->block(0, 0, firstDimension, firstDimension) = D_g_first;
      Hv->block(firstDimension, firstDimension, secondDimension,
                secondDimension) = D_h_second;
    }
    return ProductLieGroup(g, h);
  }

 public:
  /// @name Testable interface
  /// @{
  void print(const std::string& s = "") const {
    std::cout << s << "ProductLieGroup" << std::endl;
    traits<G>::Print(this->first, "  first");
    traits<H>::Print(this->second, "  second");
  }

  bool equals(const ProductLieGroup& other, double tol = 1e-9) const {
    return traits<G>::Equals(this->first, other.first, tol) &&
           traits<H>::Equals(this->second, other.second, tol);
  }
  /// @}
};

/**
 * @brief Template to construct the N-fold power of a Lie group
 * Represents the group G^N = G x G x ... x G (N times)
 * Assumes Lie group structure for G and N >= 2
 */
template <typename G, size_t N>
class PowerLieGroup : public std::array<G, N> {
  static_assert(N >= 1, "PowerLieGroup requires N >= 1");
  GTSAM_CONCEPT_ASSERT(IsLieGroup<G>);
  GTSAM_CONCEPT_ASSERT(IsTestable<G>);

 public:
  /// Base array type
  typedef std::array<G, N> Base;

 protected:
  /// Dimension of the base group
  static constexpr size_t baseDimension = traits<G>::dimension;

 public:
  /// @name Standard Constructors
  /// @{

  /// Default constructor yields identity
  PowerLieGroup() { this->fill(traits<G>::Identity()); }

  /// Construct from array of group elements
  PowerLieGroup(const Base& elements) : Base(elements) {}

  /// Construct from initializer list
  PowerLieGroup(const std::initializer_list<G>& elements) {
    if (elements.size() != N) {
      throw std::invalid_argument(
          "PowerLieGroup: initializer list size must equal N");
    }
    std::copy(elements.begin(), elements.end(), this->begin());
  }

  /// @}
  /// @name Group Operations
  /// @{

  typedef multiplicative_group_tag group_flavor;

  /// Identity element
  static PowerLieGroup Identity() { return PowerLieGroup(); }

  /// Group multiplication
  PowerLieGroup operator*(const PowerLieGroup& other) const {
    PowerLieGroup result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = traits<G>::Compose((*this)[i], other[i]);
    }
    return result;
  }

  /// Group inverse
  PowerLieGroup inverse() const {
    PowerLieGroup result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = traits<G>::Inverse((*this)[i]);
    }
    return result;
  }

  /// Compose with another element (same as operator*)
  PowerLieGroup compose(const PowerLieGroup& g) const { return (*this) * g; }

  /// Calculate relative transformation
  PowerLieGroup between(const PowerLieGroup& g) const {
    return this->inverse() * g;
  }

  /// @}
  /// @name Manifold Operations
  /// @{

  /// Manifold dimension
  static constexpr size_t dimension = N * baseDimension;

  /// Return manifold dimension
  static size_t Dim() { return dimension; }

  /// Return manifold dimension
  size_t dim() const { return dimension; }

  /// Tangent vector type
  typedef Eigen::Matrix<double, dimension, 1> TangentVector;

  /// Chart Jacobian type
  typedef OptionalJacobian<dimension, dimension> ChartJacobian;

  /// Retract to manifold
  PowerLieGroup retract(const TangentVector& v, ChartJacobian H1 = {},
                        ChartJacobian H2 = {}) const {
    if (H1 || H2) {
      throw std::runtime_error(
          "PowerLieGroup::retract derivatives not implemented yet");
    }
    PowerLieGroup result;
    for (size_t i = 0; i < N; ++i) {
      const auto vi = v.template segment<baseDimension>(i * baseDimension);
      result[i] = traits<G>::Retract((*this)[i], vi);
    }
    return result;
  }

  /// Local coordinates on manifold
  TangentVector localCoordinates(const PowerLieGroup& g, ChartJacobian H1 = {},
                                 ChartJacobian H2 = {}) const {
    if (H1 || H2) {
      throw std::runtime_error(
          "PowerLieGroup::localCoordinates derivatives not implemented yet");
    }
    TangentVector v;
    for (size_t i = 0; i < N; ++i) {
      const auto vi = traits<G>::Local((*this)[i], g[i]);
      v.template segment<baseDimension>(i * baseDimension) = vi;
    }
    return v;
  }

  /// @}
  /// @name Lie Group Operations
  /// @{

 public:
  /// Jacobian types for internal use
  typedef Eigen::Matrix<double, dimension, dimension> Jacobian;
  typedef Eigen::Matrix<double, baseDimension, baseDimension> BaseJacobian;

  /// Compose with Jacobians
  PowerLieGroup compose(const PowerLieGroup& other, ChartJacobian H1,
                        ChartJacobian H2 = {}) const {
    std::array<BaseJacobian, N> jacobians;
    PowerLieGroup result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = traits<G>::Compose((*this)[i], other[i],
                                     H1 ? &jacobians[i] : nullptr);
    }
    if (H1) {
      H1->setZero();
      for (size_t i = 0; i < N; ++i) {
        H1->template block<baseDimension, baseDimension>(
            i * baseDimension, i * baseDimension) = jacobians[i];
      }
    }
    if (H2) *H2 = Jacobian::Identity();
    return result;
  }

  /// Between with Jacobians
  PowerLieGroup between(const PowerLieGroup& other, ChartJacobian H1,
                        ChartJacobian H2 = {}) const {
    std::array<BaseJacobian, N> jacobians;
    PowerLieGroup result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = traits<G>::Between((*this)[i], other[i],
                                     H1 ? &jacobians[i] : nullptr);
    }
    if (H1) {
      H1->setZero();
      for (size_t i = 0; i < N; ++i) {
        H1->template block<baseDimension, baseDimension>(
            i * baseDimension, i * baseDimension) = jacobians[i];
      }
    }
    if (H2) *H2 = Jacobian::Identity();
    return result;
  }

  /// Inverse with Jacobian
  PowerLieGroup inverse(ChartJacobian D) const {
    std::array<BaseJacobian, N> jacobians;
    PowerLieGroup result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = traits<G>::Inverse((*this)[i], D ? &jacobians[i] : nullptr);
    }
    if (D) {
      D->setZero();
      for (size_t i = 0; i < N; ++i) {
        D->template block<baseDimension, baseDimension>(
            i * baseDimension, i * baseDimension) = jacobians[i];
      }
    }
    return result;
  }

  /// Exponential map
  static PowerLieGroup Expmap(const TangentVector& v, ChartJacobian Hv = {}) {
    std::array<BaseJacobian, N> jacobians;
    PowerLieGroup result;
    for (size_t i = 0; i < N; ++i) {
      const auto vi = v.template segment<baseDimension>(i * baseDimension);
      result[i] = traits<G>::Expmap(vi, Hv ? &jacobians[i] : nullptr);
    }
    if (Hv) {
      Hv->setZero();
      for (size_t i = 0; i < N; ++i) {
        Hv->template block<baseDimension, baseDimension>(
            i * baseDimension, i * baseDimension) = jacobians[i];
      }
    }
    return result;
  }

  /// Logarithmic map
  static TangentVector Logmap(const PowerLieGroup& p, ChartJacobian Hp = {}) {
    std::array<BaseJacobian, N> jacobians;
    TangentVector v;
    for (size_t i = 0; i < N; ++i) {
      const auto vi = traits<G>::Logmap(p[i], Hp ? &jacobians[i] : nullptr);
      v.template segment<baseDimension>(i * baseDimension) = vi;
    }
    if (Hp) {
      Hp->setZero();
      for (size_t i = 0; i < N; ++i) {
        Hp->template block<baseDimension, baseDimension>(
            i * baseDimension, i * baseDimension) = jacobians[i];
      }
    }
    return v;
  }

  /// Local coordinates (same as Logmap)
  static TangentVector LocalCoordinates(const PowerLieGroup& p,
                                        ChartJacobian Hp = {}) {
    return Logmap(p, Hp);
  }

  /// Right multiplication by exponential map
  PowerLieGroup expmap(const TangentVector& v) const {
    return compose(PowerLieGroup::Expmap(v));
  }

  /// Logarithmic map for relative transformation
  TangentVector logmap(const PowerLieGroup& g) const {
    return PowerLieGroup::Logmap(between(g));
  }

  /// Adjoint map
  Jacobian AdjointMap() const {
    Jacobian adj = Jacobian::Zero();
    for (size_t i = 0; i < N; ++i) {
      const auto adjGi = traits<G>::AdjointMap((*this)[i]);
      adj.template block<baseDimension, baseDimension>(
          i * baseDimension, i * baseDimension) = adjGi;
    }
    return adj;
  }

  /// @}

  /// @name Testable interface
  /// @{
  void print(const std::string& s = "") const {
    std::cout << s << "PowerLieGroup" << std::endl;
    for (size_t i = 0; i < N; ++i) {
      traits<G>::Print((*this)[i], "  component[" + std::to_string(i) + "]");
    }
  }

  bool equals(const PowerLieGroup& other, double tol = 1e-9) const {
    for (size_t i = 0; i < N; ++i) {
      if (!traits<G>::Equals((*this)[i], other[i], tol)) {
        return false;
      }
    }
    return true;
  }
  /// @}
};

/// Traits specialization for ProductLieGroup
template <typename G, typename H>
struct traits<ProductLieGroup<G, H>>
    : internal::LieGroup<ProductLieGroup<G, H>> {};

/// Traits specialization for PowerLieGroup
template <typename G, size_t N>
struct traits<PowerLieGroup<G, N>> : internal::LieGroup<PowerLieGroup<G, N>> {};

}  // namespace gtsam
