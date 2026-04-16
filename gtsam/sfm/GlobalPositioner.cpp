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
 * @brief GLOMAP-style joint estimation of camera and landmark positions.
 */

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/sfm/GlobalPositioner.h>

#include <stdexcept>

using namespace gtsam;
using namespace std;

namespace {
std::mt19937 kPRNG(42);
}  // namespace

NonlinearFactorGraph GlobalPositioner::buildGraph(
    const CameraPointDirections &cameraPointDirections) const {
  return LocationRecovery::buildGraph(cameraPointDirections,
                                     /*bilinear=*/true);
}

void GlobalPositioner::addPrior(
    Key anchorCameraKey, NonlinearFactorGraph *graph,
    const SharedNoiseModel &priorNoiseModel) const {
  addAnchorPrior(anchorCameraKey, graph, priorNoiseModel);
}

Values GlobalPositioner::initializeRandomly(
    const std::set<Key> &cameraKeys, const std::set<Key> &landmarkKeys,
    size_t numEdges, std::mt19937 *rng, const Values &initialValues) const {
  std::set<Key> allKeys(cameraKeys);
  allKeys.insert(landmarkKeys.begin(), landmarkKeys.end());
  return LocationRecovery::initializeRandomly(allKeys, numEdges,
                                              /*bilinear=*/true, rng,
                                              initialValues);
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
  if (cameraKeys.find(anchorCameraKey) == cameraKeys.end()) {
    throw std::invalid_argument(
        "GlobalPositioner::run: anchorCameraKey must be in cameraKeys.");
  }

  NonlinearFactorGraph graph = buildGraph(cameraPointDirections);
  addPrior(anchorCameraKey, &graph);
  Values initial = initializeRandomly(cameraKeys, landmarkKeys,
                                      cameraPointDirections.size(),
                                      initialValues);

  LevenbergMarquardtOptimizer lm(graph, initial, lmParams_);
  return lm.optimize();
}
