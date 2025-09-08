/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file Pose3Localization_StarryNight.cpp
 * @brief A 3D Stereo localization example, using during-solve and post-solve interpolation
 * @date September 2, 2025
 * @author Zi Cong Guo
 */

#include "Pose3Localization_StarryNight.h"

auto WNOAPSD = Vector6(1.0, 1.0, 1.0, 0.1, 0.1, 0.1);  // power spectral density for WNOA

size_t numPoses, numLandmarks, poseInterval;

using StateDataSet = Interpolator<Pose3>::StateDataSet;
using CovarianceMap = Interpolator<Pose3>::CovarianceMap;

std::tuple<std::string, std::string>
parseCommandLine(int argc, char** argv) {
  // Get configuration data
  std::string config_filename = "starryNight";  // default config file
  std::string output_filename = "";

  // Parse command line arguments

  // uncomment if there are bool arguments to parse
  // auto parseBool = [](const std::string& s) {
  //   if (s == "true" || s == "1") return true;
  //   if (s == "false" || s == "0") return false;
  //   std::cerr << "Invalid boolean value: '" << s << "'. Use true/false or 1/0." << std::endl;
  //   std::abort();
  // };

  auto getArg = [&](int& i) -> std::string {
    if (++i >= argc) {
      std::cerr << argv[i - 1] << " requires an argument." << std::endl;
      exit(1);
    }
    return argv[i];
  };

  std::unordered_map<std::string, std::function<void(int&)>> handlers = {
    {"--config-file",         [&](int& i) { config_filename = getArg(i); }},
    {"--output-file",       [&](int& i) { output_filename = getArg(i); }},
    {"--help",                [&](int&) {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --config-file <file>, -c <file>      Specify configuration YAML file (default: starryNight).\n"
                << "  --output-file <file>, -o <file>      Specify output filename suffix (default: derived from config file).\n"
                << "  --help, -h                           Show this help message.\n";
      exit(0);
    }},
    {"-c", [&](int& i) { handlers["--config-file"](i); }},
    {"-o", [&](int& i) { handlers["--output-file"](i); }},
    {"-h", [&](int& i) { handlers["--help"](i); }}
  };

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (handlers.count(arg)) {
      handlers[arg](i);  // handler modifies `i` if it consumes an extra argument
    } else {
      std::cerr << "Unknown argument: " << arg << "\nUse --help for usage information." << std::endl;
      exit(1);
    }
  }

  return std::make_tuple(config_filename, output_filename);
}

std::tuple<NonlinearFactorGraph, Values, Values, StateDataSet, Marginals>
optimize(const std::vector<std::pair<double, Vector6>>& inputs,
         const std::vector<MeasTriple>& measTripleVector,
         const std::vector<Pose3>& gtPoses,
         const std::vector<Point3>& gtLandmarks,
         const Cal3_S2Stereo::shared_ptr& Cal,
         const Pose3& T_cv,
         const NM::Diagonal::shared_ptr& odomNoise,
         const NM::Diagonal::shared_ptr& stereoNoise,
         const FileUtils::ConfigParams& params) {

  NonlinearFactorGraph graph;
  Values initialEstimate;
  StateDataSet stateDataSet;

  // if interp_during_solve is enabled, only add every nth state to the factor graph
  poseInterval = params.interpDuringSolve ? params.interpSkipStates : 1;
  size_t numPoses_largest_multiple = (numPoses - 1) / poseInterval * poseInterval + 1;

  // 1. ADD MOTION PRIOR

  // Add a prior on the first pose
  auto priorNoise = NM::Diagonal::Sigmas((Vector(6)<<0.003,0.003,0.003,0.001,0.001,0.001).finished());
  graph.addPrior(Symbol('x', 0), gtPoses[0], priorNoise);

  // Add odometry and/or WNOA between poses
  // WNOA is auto added later if interp_during_solve is enabled, and odometry is not
  // compatible with interp during solve (for now)
  std::cout << "Adding motion priors..." << std::endl;
  if (params.useWNOAFactor || params.useOdomFactor) {

    for (size_t poseID = 0; poseID < numPoses; ++poseID) {

      if (poseID < numPoses - 1) {
        double dt = inputs[poseID + 1].first - inputs[poseID].first;

        // add odometry factors
        if (params.useOdomFactor) {
          Vector6 odometry = inputs[poseID].second; // omega, v
          Pose3 odometryPose = Pose3::Expmap(odometry*dt);
          auto odomNoiseDiscrete = NM::Diagonal::Sigmas(odomNoise->sigmas() * dt);
          graph.emplace_shared<BetweenFactor<Pose3>>(Symbol('x', poseID), Symbol('x', poseID + 1), odometryPose, odomNoiseDiscrete);
        }

        // add WNOA motion factors
        if (params.useWNOAFactor) {
          graph.emplace_shared<WNOAMotionFactor<Pose3>>(Symbol('x', poseID), Symbol('v', poseID), Symbol('x', poseID + 1), Symbol('v', poseID + 1), dt, WNOAPSD);
        }
      }

      stateDataSet.insert(StateData(Symbol('x', poseID), Symbol('v', poseID), inputs[poseID].first));
    }
  }

  // set an initial point for poses and velocities for the optimizer
  std::cout << "Setting initial estimate..." << std::endl;
  if (params.initAtGT) {
    // Initialize poses to groundtruth
    for (size_t poseID = 0; poseID < numPoses; ++poseID) {
      initialEstimate.insert(Symbol('x', poseID), gtPoses[poseID]);
    }
    if (params.useWNOAFactor) {
      // Initialize velocities to groundtruth, if velocities exist (i.e., only if WNOA is used)
      for (size_t poseID = 0; poseID < numPoses; ++poseID) {
        initialEstimate.insert(Symbol('v', poseID), inputs[poseID].second);
      }
    }

  } else {
    // rollout odometry to initialize the poses
    auto currPredictedPose = gtPoses[0];
    initialEstimate.insert(Symbol('x', 0), currPredictedPose);
    for (size_t poseID = 0; poseID < numPoses - 1; ++poseID) {
      double dt = inputs[poseID + 1].first - inputs[poseID].first;
      Vector6 odometry = inputs[poseID].second; // v, omega
      Pose3 odometryPose = Pose3::Expmap(odometry*dt);
      currPredictedPose = currPredictedPose.compose(odometryPose);
      initialEstimate.insert(Symbol('x', poseID + 1), currPredictedPose);
    }
    if (params.useWNOAFactor) {
      // Initialize velocities to zero
      for (size_t poseID = 0; poseID < numPoses; ++poseID) {
        initialEstimate.insert(Symbol('v', poseID), Vector6(Vector6::Zero()));
      }
    }
  }

  // 2. ADD LANDMARKS

  // add all groundtruth landmark positions
  // for SLAM, maybe add a prior on the landmarks if unstable
  std::cout << "Adding landmarks..." << std::endl;
  for (size_t landmarkID = 0; landmarkID < numLandmarks; ++landmarkID) {
    if (params.fixLandmarks) {
      // use a hard constraint to fix the landmark if doing localization
      graph.add(NonlinearEquality<Point3>(Symbol('l', landmarkID), gtLandmarks[landmarkID]));
    }
    initialEstimate.insert(Symbol('l', landmarkID), gtLandmarks[landmarkID]);
  }

  // 3. ADD MEASUREMENTS
  std::cout << "Adding measurements..." << std::endl;
  if (params.useMeasurements) {    
    // Add stereo measurement factors
    for (const auto& measTriple : measTripleVector) {
      auto [poseID, landmarkID, measurement] = measTriple;
      // replace numPoses_largest_multiple with params.maxPoses once we can do extrapolation
      if (poseID < numPoses_largest_multiple && landmarkID < numLandmarks) {
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

  // 4. SET UP INTERPOLATION DURING SOLVE
  if (params.interpDuringSolve) {

    // split data into estimated and interpolated states
    StateDataSet interpDataSet;
    StateDataSet estimDataSet;
    for (size_t poseID = 0; poseID < numPoses; poseID++) {
      if (poseID % poseInterval == 0) {
        estimDataSet.insert(StateData(Symbol('x', poseID), Symbol('v', poseID), inputs[poseID].first));
      } else {
        // Remove the below condition after interpolation can also extrapolate
        if (poseID < numPoses_largest_multiple)
          interpDataSet.insert(StateData(Symbol('x', poseID), Symbol('v', poseID), inputs[poseID].first));
        // remove interpolated states from initial values - required for LevenbergMarquardt
        initialEstimate.erase(Symbol('x', poseID));
        initialEstimate.erase(Symbol('v', poseID));
      }
    }
    std::cout << "interpDataSet size: " << interpDataSet.size() << std::endl;
    std::cout << "estimDataSet size: " << estimDataSet.size() << std::endl;
    stateDataSet = estimDataSet;  // only the estimated states are part of the stateDataSet for the main solve

    // Generate interpolated version of graph
    graph = interpolateFactorGraph<Pose3>(graph, estimDataSet, interpDataSet, WNOAPSD);
  } 

  // 5. OPTIMIZE
  std::cout << "Optimizing..." << std::endl;
  LevenbergMarquardtParams parameters;
  parameters.relativeErrorTol = 1e-5;
  parameters.maxIterations = 100;
  parameters.setVerbosity("ERROR");
  Values result = LevenbergMarquardtOptimizer(graph, initialEstimate, parameters).optimize();

  // Calculate marginal covariances for the main solve
  Marginals marginals(graph, result);

  return std::make_tuple(graph, initialEstimate, result, stateDataSet, marginals);
}

// Interpolate for the in-between states not within the factor graph
std::tuple<Values, std::shared_ptr<CovarianceMap>>
interpolateAfterSolve(
  const NonlinearFactorGraph& graphMainSolve,
  const Values& resultsMainSolve,
  const StateDataSet& mainSolveStateDataSet,
  const std::vector<std::pair<double, Vector6>>& inputs,
  const FileUtils::ConfigParams& params) {

  Interpolator<Pose3> interpolator(WNOAPSD);
  std::cout << "Interpolating poses and velocities after optimization..." << std::endl;

  // Add all non-main-solve poses into the queryStateDataSet
  StateDataSet queryStateDataSet;
  // size_t numPoses_largest_multiple = (numPoses - 1) / poseInterval * poseInterval + 1;
  for (size_t poseID = 0; poseID < numPoses; ++poseID) {
    if (poseID % poseInterval != 0) {  // not a main solve pose
      auto queryTime = inputs[poseID].first;
      queryStateDataSet.insert(StateData(Symbol('x', poseID), Symbol('v', poseID), queryTime));
    }
  }
  auto covarianceMap = params.interpCovariance ? std::make_shared<CovarianceMap>() : nullptr;
  // Interpolate
  Values interpolatedValues = interpolator.interpolatePosesAndVelocities(
    graphMainSolve, resultsMainSolve, mainSolveStateDataSet, queryStateDataSet, covarianceMap);

  // Add the interpolated poses and velocities to the result
  Values resultsAll = resultsMainSolve;
  for (const auto& [key, value] : interpolatedValues) {
    resultsAll.insert(key, value);
  }

  std::cout << "Interpolated " << interpolatedValues.size() << " poses and velocities." << std::endl;
  return std::make_tuple(resultsAll, covarianceMap);
}

// --------------------------------------------------------------------------
int main(int argc, char** argv) {
  
  // Load configuration file and output filename from command line
  auto [config_filename, output_filename] = parseCommandLine(argc, argv);
  FileUtils fileUtils(config_filename, output_filename);
  auto params = fileUtils.getParams();

  // Load inputs, metadata, ground truth poses and landmarks, and measurements
  auto [inputs, gtPoses, gtLandmarks, Cal, T_cv, odomNoise, stereoNoise, measTripleVector] = fileUtils.loadAllData();

  numPoses = std::min(inputs.size(), params.maxPoses);
  numLandmarks = gtLandmarks.size();

  std::cout.precision(3);

  // Optimize the main-solve factor graph
  auto [graph, initialEstimate, result, stateDataSet, marginals] =
   optimize(inputs, measTripleVector, gtPoses, gtLandmarks, Cal, T_cv, odomNoise, stereoNoise, params);
  
  // Interpolate after the main solve if enabled
  std::shared_ptr<CovarianceMap> covarianceMap;
  if (params.interpAfterSolve) {
    std::tie(result, covarianceMap) = interpolateAfterSolve(graph, result, stateDataSet, inputs, params);
  }

  // Also package groundtruth poses and velocities for saving to file
  Values groundtruth;
  for (size_t poseID = 0; poseID < numPoses; ++poseID) {
    groundtruth.insert(Symbol('x', poseID), gtPoses[poseID]);
    groundtruth.insert(Symbol('v', poseID), inputs[poseID].second);
  }
  // Save results to files
  fileUtils.saveAllResultsToFile(
    numPoses, numLandmarks, graph, initialEstimate, result, marginals, covarianceMap, groundtruth, poseInterval);

  exit(0);
}
