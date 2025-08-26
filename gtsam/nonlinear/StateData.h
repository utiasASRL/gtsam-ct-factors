#pragma once

#include <gtsam/inference/Key.h>

#include <functional>

namespace gtsam {

/* @brief State data structure for keeping track of pose and velocity keys as
 * well as associated timestamp. Used in GP interpolation.*/
struct StateData {
  Key pose;
  Key vel;
  double time;
  // Default constructor for easy init
  StateData() = default;
  // Constructor
  StateData(Key pose_in, Key vel_in, double time_in)
      : pose(pose_in), vel(vel_in), time(time_in) {};

  // Less than operator to enable sorting
  bool operator<(const StateData& other) const {
    return this->time < other.time;
  }
  // less than operator to compare with other times
  bool operator<(double time) const { return this->time < time; }
  // equality operator for unordered sets
  //N
  bool operator==(const StateData& other) const {
    return this->pose == other.pose && this->vel == other.vel;
  }

};
}  // namespace gtsam

/* @brief Function to make StateData hashable via the pose key. This allows us
 * to make well-defined unordered sets and maps on StateData structs.
 * NOTE: hash function combines pose and vel.*/
namespace std {
template <>
struct hash<gtsam::StateData> {
  std::size_t operator()(const gtsam::StateData& p) const noexcept {
    return std::hash<uint64_t>()(p.pose) ^ (std::hash<uint64_t>()(p.vel) << 1);
  }
};
}  // namespace std