/**
 *  @file   GnssCommon.h
 *  @brief  Shared constants and utilities for GNSS factors.
 *  @date   April 14, 2026
 **/
#pragma once

#include <gtsam/base/OptionalJacobian.h>
#include <gtsam/geometry/Point3.h>

#include <limits>

namespace gtsam {
namespace gnss {

/// Speed of light in a vacuum (m/s).
constexpr double C_LIGHT = 299792458.0;

/// WGS-84 Earth rotation rate (rad/s).
constexpr double OMGE = 7.2921151467e-5;

/**
 * Geometric distance with Sagnac correction.
 *
 * Computes the Euclidean distance between a satellite and a receiver in ECEF,
 * plus the first-order Sagnac correction for the rotating Earth.  When the
 * satellite and receiver positions coincide (within machine epsilon) the unit
 * vector is set to zero and the optional Jacobian is set to zero, which avoids
 * NaNs from dividing by zero.
 *
 * @param[in]  sat   Satellite ECEF position (m).
 * @param[in]  rcv   Receiver ECEF position (m).
 * @param[out] e     Unit vector from receiver to satellite.
 * @param[out] H_rcv Optional Jacobian of the returned range w.r.t. rcv (1x3).
 *                   Includes the contribution from the Sagnac term.
 * @return     Geometric range with Sagnac correction (m).
 */
inline double geodist(const Point3& sat, const Point3& rcv, Point3& e,
                      OptionalJacobian<1, 3> H_rcv = {}) {
  const Point3 dr = sat - rcv;
  const double r = dr.norm();
  const double sagnac =
      OMGE * (sat.x() * rcv.y() - sat.y() * rcv.x()) / C_LIGHT;
  if (r < std::numeric_limits<double>::epsilon()) {
    e = Point3::Zero();
    if (H_rcv) H_rcv->setZero();
    return sagnac;
  }
  e = dr / r;
  if (H_rcv) {
    // d(||sat - rcv||)/d(rcv) = -e^T (1x3)
    // d(sagnac)/d(rcv)        = OMGE / C_LIGHT * [-sat.y(), sat.x(), 0]
    (*H_rcv)(0, 0) = -e.x() - OMGE * sat.y() / C_LIGHT;
    (*H_rcv)(0, 1) = -e.y() + OMGE * sat.x() / C_LIGHT;
    (*H_rcv)(0, 2) = -e.z();
  }
  return r + sagnac;
}

}  // namespace gnss
}  // namespace gtsam
