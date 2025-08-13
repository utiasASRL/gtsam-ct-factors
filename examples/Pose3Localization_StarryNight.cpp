/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file RangeISAMExample_plaza2.cpp
 * @brief A 2D Range SLAM example
 * @date June 20, 2013
 * @author Frank Dellaert
 */

#include <unordered_map>
#include <functional>
#include <cstdlib>
#include "Pose3Localization_StarryNight.h"

bool USE_MEASUREMENTS = true;
bool USE_WNOA_FACTOR = true;  // use WNOA factor for motion prior
bool USE_ODOM_FACTOR = false;  // use BetweenFactor for motion prior
bool INIT_AT_GT = false;  // initialize at ground truth
size_t POSE_INTERVAL_MEAS = 1;  // measurements for every nth pose is added to the factor graph
size_t MAX_POSES = 1000;  // maximum number of poses to process
bool INTERP_COVARIANCE = false;  // compute covariance of interpolated poses and velocities

auto WNOAPSD = Vector6(1.0, 1.0, 1.0, 0.1, 0.1, 0.1);  // power spectral density for WNOA

size_t numPoses, numLandmarks;

using TimestampKeyMap = std::map<double, std::pair<Key, Key>>;  // todo: don't redefine this, use the one in Interpolator.h
using CovarianceMap = std::map<Key, Matrix>;  // todo: don't redefine this, use the one in Interpolator.h

std::tuple<NonlinearFactorGraph, Values, Values, TimestampKeyMap>
optimize(const std::vector<std::pair<double, Vector6>>& inputs,
         const std::vector<MeasTriple>& measTripleVector,
         const std::vector<Pose3>& gtPoses,
         const std::vector<Point3>& gtLandmarks,
         const Cal3_S2Stereo::shared_ptr& Cal,
         const Pose3& T_cv,
         const NM::Diagonal::shared_ptr& odomNoise,
         const NM::Diagonal::shared_ptr& stereoNoise,
         bool includeIntermediatePoses) {

  NonlinearFactorGraph graph;
  Values initialEstimate;
  TimestampKeyMap timestampKeyMap;

  size_t poseInterval = includeIntermediatePoses ? 1 : POSE_INTERVAL_MEAS;

  // Add a prior on the first pose
  auto priorNoise = NM::Diagonal::Sigmas((Vector(6)<<0.003,0.003,0.003,0.001,0.001,0.001).finished());
  // // get a prior noise model for the first pose, using the first dt
  // double dt_1 = inputs[1].first - inputs[0].first;
  // auto odomNoiseDiscrete = NM::Diagonal::Sigmas(odomNoise->sigmas() * dt_1);
  graph.addPrior(Symbol('x', 0), gtPoses[0], priorNoise);

  // add factors to the graph
  if (USE_ODOM_FACTOR) {
    for (size_t poseID = 0; poseID < numPoses - poseInterval; poseID += poseInterval) {
      double dt = inputs[poseID + poseInterval].first - inputs[poseID].first;
      Vector6 odometry = inputs[poseID].second; // omega, v
      Pose3 odometryPose = Pose3::Expmap(odometry*dt);
      auto odomNoiseDiscrete = NM::Diagonal::Sigmas(odomNoise->sigmas() * dt);
      graph.emplace_shared<BetweenFactor<Pose3> >(Symbol('x', poseID), Symbol('x', poseID + poseInterval), odometryPose, odomNoiseDiscrete);
    }
  }

  if (USE_WNOA_FACTOR || USE_ODOM_FACTOR) {

    for (size_t poseID = 0; poseID < numPoses; poseID += poseInterval) {
      if (poseID < numPoses - poseInterval) {

        double dt = inputs[poseID + poseInterval].first - inputs[poseID].first;
        if (USE_ODOM_FACTOR) {
          Vector6 odometry = inputs[poseID].second; // omega, v
          Pose3 odometryPose = Pose3::Expmap(odometry*dt);
          auto odomNoiseDiscrete = NM::Diagonal::Sigmas(odomNoise->sigmas() * dt);
          graph.emplace_shared<BetweenFactor<Pose3> >(Symbol('x', poseID), Symbol('x', poseID + poseInterval), odometryPose, odomNoiseDiscrete);
        }

        if (USE_WNOA_FACTOR) {
          graph.emplace_shared<WNOAMotionFactor<Pose3> >(Symbol('x', poseID), Symbol('v', poseID), Symbol('x', poseID + poseInterval), Symbol('v', poseID + poseInterval), dt, WNOAPSD);
        }
      }

      timestampKeyMap[inputs[poseID].first] = std::pair(Symbol('x', poseID), Symbol('v', poseID));
    }
  }

  // initialize poses and velocities
  if (INIT_AT_GT) {
    // numPoses = gtPoses.size();
    for (size_t poseID = 0; poseID < numPoses; poseID += poseInterval) {
      initialEstimate.insert(Symbol('x', poseID), gtPoses[poseID]);
    }
    if (USE_WNOA_FACTOR) {
      // Initialize velocities to groundtruth
      for (size_t poseID = 0; poseID < numPoses; poseID += poseInterval) {
        initialEstimate.insert(Symbol('v', poseID), inputs[poseID].second);
      }
    }
  } else {
    // rollout odometry to initialize the poses
    auto currPredictedPose = gtPoses[0];
    initialEstimate.insert(Symbol('x', 0), currPredictedPose);
    for (size_t poseID = 0; poseID < numPoses - poseInterval; poseID += poseInterval) {
      double dt = inputs[poseID + poseInterval].first - inputs[poseID].first;
      Vector6 odometry = inputs[poseID].second; // v, omega
      Pose3 odometryPose = Pose3::Expmap(odometry*dt);
      currPredictedPose = currPredictedPose.compose(odometryPose);
      initialEstimate.insert(Symbol('x', poseID + poseInterval), currPredictedPose);
    }
    if (USE_WNOA_FACTOR) {
      // Initialize velocities to zero
      for (size_t poseID = 0; poseID < numPoses; poseID += poseInterval) {
        initialEstimate.insert(Symbol('v', poseID), Vector6(Vector6::Zero()));
      }
    }
  }

  // tried to use Expression<StereoPoint2> but it does not work
  // auto prediction = StereoPoint2_(&project2_static, T_cv_, Point3_(landmark), Cal_);

  // for localization, add all groundtruth landmark positions
  for (size_t landmarkID = 0; landmarkID < numLandmarks; ++landmarkID) {
    graph.add(NonlinearEquality<Point3>(Symbol('l', landmarkID), gtLandmarks[landmarkID]));
    initialEstimate.insert(Symbol('l', landmarkID), gtLandmarks[landmarkID]);
  }
  if (USE_MEASUREMENTS) {    
    // Add stereo measurement factors
    for (const auto& measTriple : measTripleVector) {
      auto [poseID, landmarkID, measurement] = measTriple;
      if (poseID < MAX_POSES && landmarkID < numLandmarks && poseID % POSE_INTERVAL_MEAS == 0) {
        // Add a stereo factor for the measurement
        // std::cout << "Measurement for pose " << poseID << " and landmark " << landmarkID << ": " << measurement << std::endl;
        // test out using StereoCamera
        auto T_iv = initialEstimate.at<Pose3>(Symbol('x', poseID));
        auto T_ic = T_iv.compose(T_cv);
        auto cam = StereoCamera(T_ic, Cal);
        // StereoPoint2 predicted_meas = cam.project(gtLandmarks[landmarkID]);
        // std::cout << "Predicted measurement for pose " << poseID << " and landmark " << landmarkID << ": " << predicted_meas << std::endl;
        graph.emplace_shared<GenericStereoFactor<Pose3, Point3> >(
          measurement, stereoNoise, Symbol('x', poseID), Symbol('l', landmarkID), Cal, T_cv);
      }
    }
  }

  GaussNewtonParams parameters;
  parameters.relativeErrorTol = 1e-5;
  parameters.maxIterations = 100;
  parameters.setVerbosity("ERROR");
  GaussNewtonOptimizer optimizer(graph, initialEstimate, parameters);
  Values result = optimizer.optimize();

  return std::make_tuple(graph, initialEstimate, result, timestampKeyMap);
}

// --------------------------------------------------------------------------
int main(int argc, char** argv) {

  // Parse command line arguments
  
  auto parseBool = [](const std::string& s) {
    if (s == "true" || s == "1") return true;
    if (s == "false" || s == "0") return false;
    std::cerr << "Invalid boolean value: '" << s << "'. Use true/false or 1/0." << std::endl;
    std::abort();
  };

  auto getArg = [&](int& i) -> std::string {
    if (++i >= argc) {
      std::cerr << argv[i - 1] << " requires an argument." << std::endl;
      exit(1);
    }
    return argv[i];
  };

  std::unordered_map<std::string, std::function<void(int&)>> handlers = {
    {"--use-wnoa-factor",     [&](int& i) { USE_WNOA_FACTOR      = parseBool(getArg(i)); }},
    {"--use-odom-factor",     [&](int& i) { USE_ODOM_FACTOR      = parseBool(getArg(i)); }},
    {"--use-measurements",    [&](int& i) { USE_MEASUREMENTS     = parseBool(getArg(i)); }},
    {"--pose-interval",       [&](int& i) { POSE_INTERVAL_MEAS   = std::stoul(getArg(i)); }},
    {"--interp-covariance",   [&](int& i) { INTERP_COVARIANCE    = parseBool(getArg(i)); }},
    {"--init-at-gt",          [&](int& i) { INIT_AT_GT           = parseBool(getArg(i)); }},
    {"--max-poses",           [&](int& i) { MAX_POSES            = std::stoul(getArg(i));}},
    {"--help",                [&](int&) {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --use-wnoa-factor <true|false>     Use WNOA factor for motion prior (default: true).\n"
                << "  --use-odom-factor <true|false>     Use odometry factor (default: false).\n"
                << "  --use-measurements <true|false>    Use measurements (default: true).\n"
                << "  --pose-interval <N>                Measurements for every Nth pose is added to the factor graph (default: 1).\n"
                << "  --interp-covariance <true|false>   Compute covariance of interpolated poses and velocities (default: false).\n"
                << "  --max-poses <number>               Max number of poses to process (default: 1000).\n"
                << "  --init-at-gt <true|false>          True: initialize at groundtruth. False (default): initialize at odom rollout.\n"
                << "  --help, -h                         Show this help message.\n";
      exit(0);
    }},
    {"-h", [&](int& i) { handlers["--help"](i); }}
  };

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (handlers.count(arg)) {
      handlers[arg](i);  // handler modifies `i` if it consumes an extra argument
    } else {
      std::cerr << "Unknown argument: " << arg << "\nUse --help for usage information." << std::endl;
      return 1;
    }
  }

    std::string filename_suffix = std::string("") +
      (USE_WNOA_FACTOR ? "_with_wnoa" : "_no_wnoa") +
      (USE_ODOM_FACTOR ? "_with_odom" : "_no_odom") +
      (USE_MEASUREMENTS ? "_with_meas" : "_no_meas") +
      (INIT_AT_GT ? "_gt_init" : "_odom_init") +
      "_pose_interval_" + std::to_string(POSE_INTERVAL_MEAS) + "_max_pose_" + std::to_string(MAX_POSES);
  std::string output_file_poses = "../results/starry_night_results_poses" + filename_suffix;
  std::string output_file_poses_dr = "../results/starry_night_results_poses_dr" + filename_suffix;
  std::string output_file_landmarks = "../results/starry_night_results_landmarks" + filename_suffix;
  std::string output_file_marginals = "../results/starry_night_results_marginals" + filename_suffix;
  std::string output_file_groundtruth = "../results/starry_night_groundtruth" + filename_suffix;

  // Load inputs, metadata, ground truth poses and landmarks, and measurements
  std::vector<TimedInput> inputs;
  FileUtils::readInputs(inputs);
  std::cout << "Loaded " << inputs.size() << " inputs." << std::endl;
  const auto [Cal, T_cv, odomNoise, stereoNoise] = FileUtils::getMetadata();
  std::cout << "Transform from camera to vehicle frame: " << T_cv << std::endl;
  std::cout << "Camera calibration: " << *Cal << std::endl;
  std::cout << "Odometry noise: " << odomNoise->sigmas() << std::endl;
  std::cout << "Stereo noise: " << stereoNoise->sigmas() << std::endl;
  std::cout << "Loaded stereo camera calibration and noise params." << std::endl;
  std::vector<Pose3> gtPoses;
  std::vector<Point3> gtLandmarks;
  FileUtils::readGroundTruth(gtPoses, gtLandmarks);
  std::cout << "Loaded " << gtPoses.size() << " ground truth poses and " << gtLandmarks.size() << " landmarks." << std::endl;
  std::vector<MeasTriple> measTripleVector;
  FileUtils::readMeasurements(measTripleVector, gtLandmarks.size());
  std::cout << "Loaded " << measTripleVector.size() << " range measurements." << std::endl;

  numPoses = std::min(inputs.size(), MAX_POSES);
  numLandmarks = gtLandmarks.size();
  auto [graph, initialEstimate, result, timestampKeyMap] = optimize(inputs, measTripleVector, gtPoses, gtLandmarks, Cal, T_cv, odomNoise, stereoNoise, true);

  // calculate marginal covariances
  std::cout.precision(3);
  Marginals marginals(graph, result);

  // Test out interpolation after optimization
  if (USE_WNOA_FACTOR && POSE_INTERVAL_MEAS > 1) {
    // Optimize again without adding intermediate poses into the factor graph
    auto [graph_small, initialEstimate_small, result_small, timestampKeyMap_small] = optimize(inputs, measTripleVector, gtPoses, gtLandmarks, Cal, T_cv, odomNoise, stereoNoise, false);
    
    // We want to interpolate for the in-between states not within the factor graph
    Interpolator<Pose3> interpolator(WNOAPSD);
    std::cout << "Interpolating poses and velocities after optimization..." << std::endl;

    // Add all non-main-solve poses into the queryKeyMap
    TimestampKeyMap queryKeyMap;
    size_t numPoses_largest_multiple = (numPoses - 1) / POSE_INTERVAL_MEAS * POSE_INTERVAL_MEAS + 1;
    for (size_t poseID = 0; poseID < numPoses_largest_multiple; ++poseID) {
      if (poseID % POSE_INTERVAL_MEAS != 0) {  // not a main solve pose
        auto queryTime = inputs[poseID].first;
        queryKeyMap[queryTime] = std::make_pair(Symbol('x', poseID), Symbol('v', poseID));
      }
    }
    auto covarianceMap = INTERP_COVARIANCE ? std::make_shared<CovarianceMap>() : nullptr;
    Values interpolatedValues = interpolator.interpolatePosesAndVelocities(graph_small, result_small, timestampKeyMap_small, queryKeyMap, covarianceMap);

    // Add the interpolated poses and velocities to the result
    Values results_all = result_small;
    for (const auto& [key, value] : interpolatedValues) {
      results_all.insert(key, value);
    }

    // // get the starting iterator
    // auto it = interpolatedPosesAndVelocities.begin();
    // for (size_t poseID = 0; poseID < numPoses_largest_multiple; ++poseID) {
    //   if (poseID % POSE_INTERVAL_MEAS != 0) {
    //     const auto& interpValue = it->second;
    //     result_small.insert(Symbol('x', poseID), interpValue->pose);
    //     result_small.insert(Symbol('v', poseID), interpValue->velocity);
    //     if (INTERP_COVARIANCE) {
    //       marginals.addCovariance(Symbol('x', poseID), interpValue->covariance);
    //     }
    //     it++;
    //   }
    // }

    std::cout << "Interpolated " << interpolatedValues.size() << " poses and velocities." << std::endl;

    FileUtils::savePosesToFile(results_all, numPoses_largest_multiple, output_file_poses + "_interpolated", 1);
    if (INTERP_COVARIANCE) {
      Marginals marginals_small(graph_small, results_all);
      std::cout << "Constructed marginals for main-solve component of interpolation." << std::endl;
      FileUtils::saveMarginalsToFile(marginals_small, numPoses_largest_multiple, output_file_marginals + "_interpolated", 1, covarianceMap);
    }
  }

  // Write to a .dot file
  graph.saveGraph("graph.dot", result);
  writeG2o(graph, initialEstimate, "starry_night_initial.g2o");
  writeG2o(graph, result, "starry_night_optimized.g2o");

  // Save results to files
  std::cout << "Saving results to files..." << std::endl;
  FileUtils::savePosesToFile(result, numPoses, output_file_poses);
  FileUtils::savePosesToFile(initialEstimate, numPoses, output_file_poses_dr);
  FileUtils::saveLandmarksToFile(result, numLandmarks, output_file_landmarks);
  FileUtils::saveMarginalsToFile(marginals, numPoses, output_file_marginals);
  // Save the ground truth poses to a file
  Values groundtruth;
  for (size_t poseID = 0; poseID < numPoses; ++poseID) {
    groundtruth.insert(Symbol('x', poseID), gtPoses[poseID]);
  }
  FileUtils::savePosesToFile(groundtruth, numPoses, output_file_groundtruth);

  exit(0);
}
