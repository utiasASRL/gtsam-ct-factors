/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010-2020, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

#pragma once

/**
 * @file GlobalPositioner.h
 * @author Kathir Gounder
 * @date 2026
 * @brief GLOMAP-style joint estimation of camera positions and 3D landmark
 *        positions from camera-to-point direction measurements.
 */

#include <gtsam/geometry/Unit3.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/sfm/BinaryMeasurement.h>

#include <random>
#include <set>
#include <vector>

namespace gtsam {

/**
 * @brief Joint estimation of camera positions and 3D landmark positions from
 * camera-to-point direction measurements (GLOMAP-style global positioning).
 *
 * Unlike TranslationRecovery (which solves the camera-camera translation
 * synchronization problem), GlobalPositioner solves a bipartite estimation
 * problem with two distinct kinds of unknowns:
 *   - Camera positions  c_i \in R^3  (from camera key set)
 *   - Landmark positions X_k \in R^3 (from landmark key set)
 *
 * The graph is structurally bipartite: every measurement is a unit direction
 * from one camera to one landmark, expressed in the world frame. Cameras
 * never directly measure other cameras and landmarks never directly measure
 * other landmarks. This is fundamentally different from synchronization
 * problems like rotation averaging or translation averaging, where all nodes
 * live in the same space and edges connect nodes of the same kind.
 *
 * The measurement equation, following BATA (Zhuang et al., CVPR 2018) and
 * GLOMAP (Pan et al., ECCV 2024), is:
 *
 *    w_iZk = Unit3(s_ik * (X_k - c_i))
 *
 * where w_iZk is the measured world-frame direction from camera i to landmark
 * k, and s_ik is a per-observation scale variable that is jointly optimized.
 * The bilinear residual is bounded, which gives the optimization a smooth
 * landscape that tolerates random initialization (in contrast to the
 * unbounded reprojection cost which requires careful initialization).
 *
 * Gauge freedom: the problem has 3 translation DOF (overall position of the
 * reconstruction) and 1 scale DOF (overall size of the reconstruction). We
 * fix translation by anchoring one camera to the origin via a strong prior.
 * We do *not* fix scale by pinning a landmark; instead, the BATA per-
 * observation scale variables establish scale through the consensus of all
 * observations. This is how GLOMAP handles gauge and is different from
 * TranslationRecovery's strategy of pinning the second key of the first edge.
 *
 * Internally uses BilinearAngleTranslationFactor (the existing GTSAM factor
 * that already implements the BATA residual). No new factor types are
 * introduced by this class.
 *
 * Relationship to TranslationRecovery: TranslationRecovery solves
 * translation synchronization (camera-camera relative translation directions
 * to camera positions). GlobalPositioner solves the bipartite camera+landmark
 * estimation problem. The two classes share the same underlying factor
 * (BilinearAngleTranslationFactor when TranslationRecovery is configured
 * with use_bilinear_translation_factor=true) but the fundamental graph structure,
 * gauge handling, and actual use-case in a full structure-from-motion pipeline is different.
 *
 * Reference: Pan, L., Barath, D., Pollefeys, M., Schonberger, J.L.,
 * "Global Structure-from-Motion Revisited," ECCV 2024.
 */
class GTSAM_EXPORT GlobalPositioner {
 public:
  using CameraPointDirection = BinaryMeasurement<Unit3>;
  using CameraPointDirections = std::vector<CameraPointDirection>;

 private:
  // Parameters for the underlying Levenberg-Marquardt optimizer.
  LevenbergMarquardtParams lmParams_;

 public:
  /**
   * @brief Construct a GlobalPositioner with optimizer parameters.
   * @param lmParams Levenberg-Marquardt parameters for the inner optimization.
   */
  explicit GlobalPositioner(const LevenbergMarquardtParams &lmParams)
      : lmParams_(lmParams) {}

  /**
   * @brief Default constructor with default LM parameters.
   */
  GlobalPositioner() = default;

  /**
   * @brief Build the factor graph for joint camera+landmark optimization.
   *
   * Each input edge contributes one BilinearAngleTranslationFactor connecting
   * a camera key, a landmark key, and a per-observation scale variable
   * (Symbol 'S', i) where i is the edge index.
   *
   * @param cameraPointDirections  unit direction measurements from cameras to
   *                  landmarks, expressed in the world frame. Each edge's key1
   *                  must be a camera key and key2 must be a landmark key.
   * @return NonlinearFactorGraph containing one BATA factor per edge.
   */
  NonlinearFactorGraph buildGraph(
      const CameraPointDirections &cameraPointDirections) const;

  /**
   * @brief Add a gauge-fixing prior on a single anchor camera.
   *
   * Pins the specified anchor camera to the origin via a strong prior on its
   * Point3 position. This fixes the 3 translation DOF of the reconstruction.
   *
   * Scale is *not* fixed by a landmark prior. Scale is established jointly
   * by the BATA per-observation scale variables (one Vector1 per edge) which
   * are initialized to 1.0 and optimized with the rest of the variables.
   * The consensus of these scale variables across all observations determines
   * the overall scale of the reconstruction.
   *
   * @param anchorCameraKey  the camera key to anchor at the origin.
   * @param graph            factor graph to which the prior is added.
   * @param priorNoiseModel  noise model for the anchor prior (default: tight
   *                         isotropic to make it effectively a hard constraint).
   */
  void addPrior(Key anchorCameraKey, NonlinearFactorGraph *graph,
                const SharedNoiseModel &priorNoiseModel =
                    noiseModel::Isotropic::Sigma(3, 1e-3)) const;

  /**
   * @brief Random initialization of camera positions, landmark positions, and
   * BATA scale variables.
   *
   * Camera and landmark positions are initialized uniformly in [-1, 1]^3,
   * matching GLOMAP's initialization-free claim. BATA scale variables (one
   * per edge) are initialized to 1.0. The bounded BATA residual makes this
   * random initialization sufficient in practice.
   *
   * @param cameraKeys     set of camera keys to initialize.
   * @param landmarkKeys   set of landmark keys to initialize.
   * @param numEdges       number of direction edges (used to create scale vars).
   * @param rng            random number generator.
   * @param initialValues  optional partial initial values to seed specific keys.
   * @return Values containing initial estimates for all unknowns.
   */
  Values initializeRandomly(const std::set<Key> &cameraKeys,
                            const std::set<Key> &landmarkKeys, size_t numEdges,
                            std::mt19937 *rng,
                            const Values &initialValues = Values()) const;

  /// Version with a fixed default RNG seed.
  Values initializeRandomly(const std::set<Key> &cameraKeys,
                            const std::set<Key> &landmarkKeys, size_t numEdges,
                            const Values &initialValues = Values()) const;

  /**
   * @brief Build the factor graph, fix gauge, and run LM optimization.
   *
   * @param cameraPointDirections  unit direction measurements from cameras to
   *                               landmarks in world frame.
   * @param cameraKeys       explicit set of camera keys.
   * @param landmarkKeys     explicit set of landmark keys.
   * @param anchorCameraKey  camera key to anchor at the origin (must be in
   *                         cameraKeys).
   * @param initialValues    optional partial initial values; missing keys are
   *                         initialized randomly.
   * @return Values containing optimized camera positions (Point3), landmark
   *         positions (Point3), and BATA scale variables (Vector1).
   */
  Values run(const CameraPointDirections &cameraPointDirections,
             const std::set<Key> &cameraKeys,
             const std::set<Key> &landmarkKeys, Key anchorCameraKey,
             const Values &initialValues = Values()) const;
};

}  // namespace gtsam
