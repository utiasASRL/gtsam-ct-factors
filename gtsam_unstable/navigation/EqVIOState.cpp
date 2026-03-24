/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file EqVIOState.cpp
 * @brief Dynamic EqVIO manifold state implementation.
 * @author Rohan Bansal
 */

#include <gtsam_unstable/navigation/EqVIOState.h>

namespace gtsam {
namespace eqvio {

/// Print one landmark block with id and world coordinates.
void Landmark::print(const std::string& s) const {
  if (!s.empty()) std::cout << s << " ";
  std::cout << "Landmark{id=" << id << ", p=" << p.transpose() << "}"
            << std::endl;
}

/// Compare landmark id and coordinates within absolute tolerance.
bool Landmark::equals(const Landmark& other, double tol) const {
  return id == other.id && equal_with_abs_tol(p, other.p, tol);
}

/// Gravity direction in body frame derived from current body pose.
Vector3 SensorState::gravityDir() const {
  return pose.rotation().unrotate(Vector3::UnitZ());
}

/// Print sensor block for debugging and test diagnostics.
void SensorState::print(const std::string& s) const {
  if (!s.empty()) std::cout << s << std::endl;
  gtsam::print(Vector(inputBias.vector()), "  inputBias");
  pose.print("  pose");
  gtsam::print(Vector(velocity), "  velocity");
  cameraOffset.print("  cameraOffset");
}

/// Compare all sensor subcomponents within tolerance.
bool SensorState::equals(const SensorState& other, double tol) const {
  return inputBias.equals(other.inputBias, tol) &&
         pose.equals(other.pose, tol) &&
         equal_with_abs_tol(velocity, other.velocity, tol) &&
         cameraOffset.equals(other.cameraOffset, tol);
}

State::State(const SensorState& sensor_, const std::vector<Landmark>& lms)
    : sensor(sensor_), cameraLandmarks(lms) {}

/// Number of landmarks in dynamic state tail.
size_t State::n() const { return cameraLandmarks.size(); }

/// Total state chart dimension.
int State::dim() const {
  return SensorState::CompDim + Landmark::CompDim * static_cast<int>(n());
}

/// Landmark ids in contiguous storage order.
std::vector<int> State::ids() const {
  std::vector<int> out;
  out.reserve(cameraLandmarks.size());
  for (const Landmark& lm : cameraLandmarks) out.push_back(lm.id);
  return out;
}

/**
 * @brief Retract state by tangent increment in EqVIO chart coordinates.
 *
 * Sensor manifold components use their native retract operations, while
 * landmark position blocks are additive.
 */
State State::retract(const TangentVector& v, ChartJacobian H1,
                     ChartJacobian H2) const {
  const int d = dim();

  Matrix66 Hpose1, Hpose2, Hcam1, Hcam2;
  State out(*this);

  out.sensor.inputBias = sensor.inputBias.retract(v.segment<6>(0));
  out.sensor.pose =
      sensor.pose.retract(v.segment<6>(6), H1 ? &Hpose1 : nullptr,
                          H2 ? &Hpose2 : nullptr);
  out.sensor.velocity += v.segment<3>(12);
  out.sensor.cameraOffset =
      sensor.cameraOffset.retract(v.segment<6>(15), H1 ? &Hcam1 : nullptr,
                                  H2 ? &Hcam2 : nullptr);

  for (size_t i = 0; i < n(); ++i) {
    out.cameraLandmarks[i].p +=
        v.segment<3>(SensorState::CompDim + 3 * static_cast<int>(i));
  }

  if (H1) {
    H1->setZero(d, d);
    H1->block<6, 6>(0, 0).setIdentity();
    H1->block<6, 6>(6, 6) = Hpose1;
    H1->block<3, 3>(12, 12).setIdentity();
    H1->block<6, 6>(15, 15) = Hcam1;
    for (size_t i = 0; i < n(); ++i) {
      const int row = SensorState::CompDim + 3 * static_cast<int>(i);
      H1->block<3, 3>(row, row).setIdentity();
    }
  }

  if (H2) {
    H2->setZero(d, d);
    H2->block<6, 6>(0, 0).setIdentity();
    H2->block<6, 6>(6, 6) = Hpose2;
    H2->block<3, 3>(12, 12).setIdentity();
    H2->block<6, 6>(15, 15) = Hcam2;
    for (size_t i = 0; i < n(); ++i) {
      const int row = SensorState::CompDim + 3 * static_cast<int>(i);
      H2->block<3, 3>(row, row).setIdentity();
    }
  }

  return out;
}

/**
 * @brief Compute chart local coordinates from this state to `other`.
 *
 * Sensor manifold components use native local coordinates, while landmark
 * position blocks are simple differences.
 */
State::TangentVector State::localCoordinates(const State& other,
                                             ChartJacobian H1,
                                             ChartJacobian H2) const {
  const int d = dim();

  Matrix66 Hpose1, Hpose2, Hcam1, Hcam2;
  TangentVector out = Vector::Zero(d);

  out.segment<6>(0) = sensor.inputBias.localCoordinates(other.sensor.inputBias);
  out.segment<6>(6) = sensor.pose.localCoordinates(
      other.sensor.pose, H1 ? &Hpose1 : nullptr, H2 ? &Hpose2 : nullptr);
  out.segment<3>(12) = other.sensor.velocity - sensor.velocity;
  out.segment<6>(15) = sensor.cameraOffset.localCoordinates(
      other.sensor.cameraOffset, H1 ? &Hcam1 : nullptr, H2 ? &Hcam2 : nullptr);

  for (size_t i = 0; i < n(); ++i) {
    out.segment<3>(SensorState::CompDim + 3 * static_cast<int>(i)) =
        other.cameraLandmarks[i].p - cameraLandmarks[i].p;
  }

  if (H1) {
    H1->setZero(d, d);
    H1->block<6, 6>(0, 0) = -I_6x6;
    H1->block<6, 6>(6, 6) = Hpose1;
    H1->block<3, 3>(12, 12) = -I_3x3;
    H1->block<6, 6>(15, 15) = Hcam1;
    for (size_t i = 0; i < n(); ++i) {
      const int row = SensorState::CompDim + 3 * static_cast<int>(i);
      H1->block<3, 3>(row, row) = -I_3x3;
    }
  }

  if (H2) {
    H2->setZero(d, d);
    H2->block<6, 6>(0, 0).setIdentity();
    H2->block<6, 6>(6, 6) = Hpose2;
    H2->block<3, 3>(12, 12).setIdentity();
    H2->block<6, 6>(15, 15) = Hcam2;
    for (size_t i = 0; i < n(); ++i) {
      const int row = SensorState::CompDim + 3 * static_cast<int>(i);
      H2->block<3, 3>(row, row).setIdentity();
    }
  }

  return out;
}

/// Print full state for debugging.
void State::print(const std::string& s) const {
  if (!s.empty()) std::cout << s << std::endl;
  std::cout << "State(dim=" << dim() << ", n=" << n() << ")" << std::endl;
  sensor.print("  sensor");
  for (size_t i = 0; i < cameraLandmarks.size(); ++i) {
    cameraLandmarks[i].print("  landmark[" + std::to_string(i) + "]");
  }
}

/// Compare sensor and landmark blocks with tolerance.
bool State::equals(const State& other, double tol) const {
  if (cameraLandmarks.size() != other.cameraLandmarks.size()) return false;
  if (!sensor.equals(other.sensor, tol)) return false;
  for (size_t i = 0; i < cameraLandmarks.size(); ++i) {
    if (!cameraLandmarks[i].equals(other.cameraLandmarks[i], tol)) return false;
  }
  return true;
}

}  // namespace eqvio
}  // namespace gtsam
