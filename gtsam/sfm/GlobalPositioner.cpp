/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010-2020, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file GlobalPositioner.cpp
 * @author Kathir Gounder
 * @date 2026
 * @brief GLOMAP-style joint estimation of camera positions and 3D landmark
 *        positions from camera-to-point direction measurements.
 */

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/sfm/GlobalPositioner.h>
#include <gtsam/sfm/TranslationFactor.h>  // for BilinearAngleTranslationFactor
#include <gtsam/slam/PriorFactor.h>

#include <random>
#include <stdexcept>

using namespace gtsam;
using namespace std;

namespace {
// Default RNG for the wrapper-friendly initializeRandomly overload, mirroring
// TranslationRecovery's pattern. Using a fixed seed makes results reproducible.
std::mt19937 kPRNG(42);

// Convert a Unit3 noise model (2D, on the manifold) to a Point3 noise model
// (3D, in the ambient space). BilinearAngleTranslationFactor's residual lives
// in R^3 (it computes scale * (Xk - ci) - measurement which is a Point3
// difference), so we need a 3D noise model.
//
// The input direction measurements come with 2D noise on the Unit3 manifold,
// but the factor's residual is 3D in the ambient space.
SharedNoiseModel convertDirectionNoiseModel(
    const SharedNoiseModel &unit3NoiseModel) {
  using noiseModel::Isotropic;
  if (auto isotropic =
          std::dynamic_pointer_cast<Isotropic>(unit3NoiseModel)) {
    return noiseModel::Isotropic::Sigma(3, isotropic->sigma());
  }
  if (auto robust = std::dynamic_pointer_cast<noiseModel::Robust>(
          unit3NoiseModel)) {
    return noiseModel::Robust::Create(robust->robust(),
                                      convertDirectionNoiseModel(robust->noise()));
  }
  throw std::runtime_error(
      "GlobalPositioner::convertDirectionNoiseModel: only isotropic "
      "(optionally robust-wrapped) noise models are supported.");
}
}  // namespace

NonlinearFactorGraph GlobalPositioner::buildGraph(
    const CameraPointDirections &cameraPointDirections) const {
  NonlinearFactorGraph graph;

  // One BilinearAngleTranslationFactor per camera-to-point direction.
  // Each factor connects: camera key (key1), landmark key (key2), and a
  // per-observation scale variable (Symbol 'S', edge_index).
  uint64_t i = 0;
  for (const auto &edge : cameraPointDirections) {
    auto model = convertDirectionNoiseModel(edge.noiseModel());
    graph.emplace_shared<BilinearAngleTranslationFactor>(
        edge.key1(), edge.key2(), Symbol('S', i), edge.measured(), model);
    ++i;
  }
  return graph;
}

void GlobalPositioner::addPrior(
    Key anchorCameraKey, NonlinearFactorGraph *graph,
    const SharedNoiseModel &priorNoiseModel) const {
  // Pin the anchor camera to the origin to fix the 3 translation DOF.
  // Scale is *not* fixed here — the BATA per-observation scale variables
  // handle scale through their joint optimization.
  graph->addPrior<Point3>(anchorCameraKey, Point3(0, 0, 0), priorNoiseModel);
}

Values GlobalPositioner::initializeRandomly(
    const std::set<Key> &cameraKeys, const std::set<Key> &landmarkKeys,
    size_t numEdges, std::mt19937 *rng, const Values &initialValues) const {
  uniform_real_distribution<double> randomVal(-1, 1);
  Values initial;

  // Helper: insert a random Point3 if the key isn't already in initialValues.
  auto insertPosition = [&](Key key) {
    if (initialValues.exists(key)) {
      initial.insert<Point3>(key, initialValues.at<Point3>(key));
    } else {
      initial.insert<Point3>(
          key, Point3(randomVal(*rng), randomVal(*rng), randomVal(*rng)));
    }
  };

  // Initialize all camera positions.
  for (const Key &cameraKey : cameraKeys) {
    insertPosition(cameraKey);
  }

  // Initialize all landmark positions.
  for (const Key &landmarkKey : landmarkKeys) {
    insertPosition(landmarkKey);
  }

  // Initialize per-observation BATA scale variables to 1.0.
  // One scale variable per edge, named with Symbol('S', edge_index).
  for (size_t i = 0; i < numEdges; ++i) {
    initial.insert<Vector1>(Symbol('S', i), Vector1(1.0));
  }

  return initial;
}

Values GlobalPositioner::initializeRandomly(
    const std::set<Key> &cameraKeys, const std::set<Key> &landmarkKeys,
    size_t numEdges, const Values &initialValues) const {
  return initializeRandomly(cameraKeys, landmarkKeys, numEdges, &kPRNG,
                            initialValues);
}

Values GlobalPositioner::run(
    const CameraPointDirections &cameraPointDirections,
    const std::set<Key> &cameraKeys, const std::set<Key> &landmarkKeys,
    Key anchorCameraKey, const Values &initialValues) const {
  // Sanity check: the anchor must actually be in the camera key set.
  if (cameraKeys.find(anchorCameraKey) == cameraKeys.end()) {
    throw std::invalid_argument(
        "GlobalPositioner::run: anchorCameraKey must be in cameraKeys.");
  }

  // Build the BATA factor graph from camera-to-point direction measurements.
  NonlinearFactorGraph graph = buildGraph(cameraPointDirections);

  // Anchor one camera at the origin to fix translation gauge.
  // Scale gauge is handled by the BATA scale variables (no landmark prior).
  addPrior(anchorCameraKey, &graph);

  // Random initialization for cameras, landmarks, and BATA scale variables.
  Values initial = initializeRandomly(cameraKeys, landmarkKeys,
                                      cameraPointDirections.size(),
                                      initialValues);

  // Run Levenberg-Marquardt.
  LevenbergMarquardtOptimizer lm(graph, initial, lmParams_);
  return lm.optimize();
}
