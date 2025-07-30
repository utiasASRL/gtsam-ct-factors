#include <gtsam/geometry/Point1.h>
#include <cmath>
#include <iostream>



using namespace std;

namespace gtsam {

/* ************************************************************************* */
double norm1(const Point1& p, OptionalJacobian<1,1> H) {
  double r = std::abs(p.x());
  if (H) {
    if (std::abs(p.x()) > 1e-10)
      *H << (p.x() >= 0 ? 1.0 : -1.0);
    else
      *H << 1.0;  // derivative of abs(x) at x=0 is undefined, using 1 as convention
  }
  return r;
}

/* ************************************************************************* */
double distance1(const Point1& p, const Point1& q, OptionalJacobian<1, 1> H1,
                 OptionalJacobian<1, 1> H2) {
  Point1 d = q - p;
  if (H1 || H2) {
    Matrix11 H;
    double r = norm1(d, H);
    if (H1) *H1 = -H;
    if (H2) *H2 =  H;
    return r;
  } else {
    return std::abs(d.x());
  }
}

/* ************************************************************************* */
ostream &operator<<(ostream &os, const gtsam::Point1Pair &p) {
  os << p.first << " <-> " << p.second;
  return os;
}

} // namespace gtsam
