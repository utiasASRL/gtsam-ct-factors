/**
 *  @file   GnssCommon.h
 *  @brief  Shared constants and utilities for GNSS factors.
 *  @date   April 14, 2026
 **/
#pragma once

#include <gtsam/geometry/Point3.h>

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
 * plus the first-order Sagnac correction for the rotating Earth.
 *
 * @param[in]  sat  Satellite ECEF position (m).
 * @param[in]  rcv  Receiver ECEF position (m).
 * @param[out] e    Unit vector from receiver to satellite.
 * @return     Geometric range with Sagnac correction (m).
 */
inline double geodist(const Point3& sat, const Point3& rcv, Point3& e) {
  const Point3 dr = sat - rcv;
  const double r = dr.norm();
  e = dr / r;
  return r + OMGE * (sat.x() * rcv.y() - sat.y() * rcv.x()) / C_LIGHT;
}

}  // namespace gnss
}  // namespace gtsam
