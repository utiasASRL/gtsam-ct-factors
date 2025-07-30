#pragma once

#include <gtsam/base/VectorSpace.h>
#include <gtsam/base/Manifold.h>
#include <gtsam/base/OptionalJacobian.h>
#include <gtsam/base/std_optional_serialization.h>
#if GTSAM_ENABLE_BOOST_SERIALIZATION
#include <boost/serialization/nvp.hpp>
#endif

#include <optional>
#include <vector>
#include <utility>
#include <iostream>

namespace gtsam {

  typedef Vector1 Point1;

  // Convenience typedef
  using Point1Pair = std::pair<Point1, Point1>;
  GTSAM_EXPORT std::ostream &operator<<(std::ostream &os, const gtsam::Point1Pair &p);

  using Point1Pairs = std::vector<Point1Pair>;

  // Distance of the point from the origin, with Jacobian
  GTSAM_EXPORT double norm1(const Point1& p, OptionalJacobian<1, 1> H = {});

  // distance between two points
  GTSAM_EXPORT double distance1(const Point1& p1, const Point1& q,
          OptionalJacobian<1, 1> H1 = {},
          OptionalJacobian<1, 1> H2 = {});

  template <typename A1, typename A2>
  struct Range;

  template <>
  struct Range<Point1, Point1> {
    typedef double result_type;
    double operator()(const Point1& p, const Point1& q,
            OptionalJacobian<1, 1> H1 = {},
            OptionalJacobian<1, 1> H2 = {}) {
      return distance1(p, q, H1, H2);
    }
  };

} // \ namespace gtsam

