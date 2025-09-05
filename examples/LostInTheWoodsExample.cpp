#include "LostInTheWoodsExample.h"

int main(int argc, char* argv[]) {
  // Get configuration data
  string config_file = "examples/Data/LostInTheWoods.yaml";
  if (argc > 1) {
    config_file = argv[1];
  }
  YAML::Node config = YAML::LoadFile(config_file);

  // Load Files
  string input_file = config["files"]["input"].as<string>();
  string output_file = config["files"]["output"].as<string>();
  string gt_output_file = config["files"]["gt_out"].as<string>();
  string interp_raw_file = config["files"]["interp_raw_file"].as<string>();
  string interp_out = config["files"]["interp_out"].as<string>();
  // Load dataset
  DatasetLoader data;
  data.loadFromFile(input_file);
  data.checkSizes();

  // switches for factors/init
  bool include_prior = config["flags"]["prior"].as<bool>();
  bool include_odom = config["flags"]["odom"].as<bool>();
  bool include_wnoa = config["flags"]["wnoa"].as<bool>();
  bool include_br_meas = config["flags"]["br"].as<bool>();
  bool gt_init = config["flags"]["gt_init"].as<bool>();
  bool solve_slam = config["flags"]["solve_slam"].as<bool>();
  // interpolation
  bool interp_enable = config["interp"]["enable"].as<bool>();
  uint interp_period = config["interp"]["interp_period"].as<uint>();
  bool fixed_noise = config["interp"]["fixed_noise"].as<bool>();
  if (interp_enable) {
    output_file = interp_out;
  }
  // Get inputs from param file
  double r_max = config["params"]["r_max"].as<double>();
  double del_t = config["params"]["del_t"].as<double>();
  int start = config["params"]["start"].as<int>();
  int end = config["params"]["end"].as<int>();
  // Get noise model parameters
  Vector sigma_prior =
      Vector3(config["noise"]["prior"].as<vector<double>>().data());
  Vector sigma_wnoa =
      Vector3(config["noise"]["wnoa"].as<vector<double>>().data());
  double sigma_y_odom = config["noise"]["odom_y"].as<double>();
  double mult_bearing = config["noise"]["bearing"].as<double>();
  double mult_range = config["noise"]["range"].as<double>();
  Vector sigma_odom =
      Vector3(sqrt(data.v_var), sigma_y_odom, sqrt(data.om_var)) * del_t;
  Vector sigma_br =
      Vector2(sqrt(mult_bearing * data.b_var), sqrt(mult_range * data.r_var));

  // Generate noise models
  auto priorNoise = noiseModel::Diagonal::Sigmas(sigma_prior);  // prior
  auto odoNoise = noiseModel::Diagonal::Sigmas(sigma_odom);     // odometry
  auto measNoise =
      noiseModel::Diagonal::Sigmas(sigma_br);  // range-bearing noise

  // Get ground truth solution
  Values gt;
  vector<StateData> all_states;
  for (int i = start; i <= end; i++) {
    gt.insert(Symbol('x', i),
              Pose2(data.x_true[i], data.y_true[i], data.th_true[i]));
    if (include_wnoa || interp_enable) {
      gt.insert(Symbol('v', i), Vector3(data.v[i], 0.0, data.om[i]));
      // create vector of states for interpolation
      all_states.push_back(
          StateData(Symbol('x', i), Symbol('v', i), data.t[i]));
    }
  }
  // Ground truth for landmarks
  for (int j = 0; j < data.n_landmarks; j++) {
    gt.insert(Symbol('l', j),
              Point2(data.landmarks(j, 0), data.landmarks(j, 1)));
  }

  // Create a factor graph
  ExpressionFactorGraph graph;

  // Starting point
  Pose2 startPose(data.x_true[start], data.y_true[start], data.th_true[start]);
  // Initial Pose Prior
  if (include_prior) {
    cout << "Adding Prior on start pose: " << sigma_prior << endl;
    graph.add(PriorFactor<Pose2>(Symbol('x', start), startPose, priorNoise));
    if (include_wnoa || interp_enable) {
      // Add in velocity prior on first state
      cout << "Adding Prior on start velocity" << endl;
      Vector vel_init = Vector3(data.v[start], 0.0, data.om[start]);
      graph.addPrior<Vector3>(Symbol('v', start), vel_init, odoNoise);
    }
  }
  // Odometry factors
  if (include_odom) {
    cout << "Adding odometry prior factors " << endl;

    for (int i = start + 1; i <= end; i++) {
      // define odometry measurement
      Pose2 odom(data.v[i - 1] * del_t, 0.0, data.om[i - 1] * del_t);
      // add factor to graph
      const auto factor = BetweenFactor<Pose2>(Symbol('x', i - 1),
                                               Symbol('x', i), odom, odoNoise);
      graph.add(factor);
    }
  }

  // White-Noise-On-Acceleration Prior
  // Only add if not adding later for interpolated factors
  if (include_wnoa) {
    cout << "Adding WNOA factors" << endl;
    // Add WNOA Motion Factors between states
    for (uint i = 0; i < all_states.size() - 1; i++) {
      graph.add(WNOAMotionFactor<Pose2>(all_states[i], all_states[i + 1],
                                        sigma_wnoa));
    }
  }

  // BearingRange Measurements
  // Create a list of 18 booleans that track which landmarks have been observed
  // at least once
  vector<bool> landmark_observed(data.n_landmarks, false);
  if (include_br_meas) {
    cout << "Adding bearing range measurement factors" << endl;

    // Define landmarks
    vector<Point2> landmarks(data.n_landmarks);
    for (int j = 0; j < data.n_landmarks; j++) {
      landmarks[j] = data.landmarks.row(j);
    }

    Pose2 T_vs(data.d, 0.0, 0.0);
    for (int i = start; i <= end; i++) {
      // Define Key
      Key xi = Symbol('x', i);
      for (int j = 0; j < data.n_landmarks; j++) {
        Key landmark = Symbol('l', j);
        // Check if we have a valid measurement
        if ((data.range(i, j) > 0.0) && (abs(data.bearing(i, j)) > 0.0) &&
            (data.range(i, j) < r_max)) {
          // Landmark has been observed
          landmark_observed[j] = true;
          // Get Bearing Range measurement
          BearingRange2 measurement(Rot2(data.bearing(i, j)), data.range(i, j));
          // Compute the bearing and range Prediction
          // If we solve slam, use unknown landmark variable, otherwise use
          // ground-truth value
          auto predict =
              solve_slam
                  ? BearingRangeLandmarkPredictionSLAM(xi, landmark, T_vs)
                  : BearingRangeLandmarkPrediction(xi, landmarks[j], T_vs);
          // Define Factor
          graph.addExpressionFactor(predict, measurement, measNoise);
        }
      }
    }
  }

  // Initialization
  Values initial;
  if (gt_init) {
    cout << "Ground truth initialization enabled" << endl;
    initial = gt;
  } else {
    cout << "Ground truth initialization disabled" << endl;
    // Rollout odometry
    for (int i = start; i <= end; i++) {
      if (i == start) {
        initial.insert(Symbol('x', i), startPose);
        Vector3 zero = Vector3::Zero();
        if (include_wnoa || interp_enable) {
          initial.insert(Symbol('v', i), zero);
        }
      } else {
        Vector vel = Vector3(data.v[i - 1], 0.0, data.om[i - 1]);
        Vector3 vel_t = del_t * vel;
        Pose2 odom = Pose2::Expmap(vel_t);
        initial.insert(Symbol('x', i),
                       initial.at<Pose2>(Symbol('x', i - 1)).compose(odom));
        if (include_wnoa || interp_enable) {
          initial.insert(Symbol('v', i), vel);
        }
      }
    }
  }

  // Initialize landmarks if doing full SLAM
  if (solve_slam) {
    // Initialize landmarks at zero
    for (int j = 0; j < data.n_landmarks; j++) {
      // Only add keys for landmarks that have been observed
      if (landmark_observed[j]) {
        initial.insert(Symbol('l', j), Point2(0, 0));
      }
    }
  }

  // set up optimizer
  LevenbergMarquardtParams params;
  params.setVerbosityLM("SUMMARY");

  // Run optimizer
  Values result;
  Values result_interp;
  if (interp_enable) {
    cout << "Interpolation enabled!" << endl;
    // process states into estimated and interpolated
    set<StateData> interp;
    set<StateData> estim;
    for (size_t i = 0; i < all_states.size(); i++) {
      if (i == 0 || i == all_states.size() - 1 || i % interp_period == 0) {
        estim.insert(all_states[i]);
      } else {
        interp.insert(all_states[i]);
        // remove interpolated states from initial values
        initial.erase(all_states[i].pose);
        initial.erase(all_states[i].vel);
      }
    }
    // Generate interpolated version of graph
    NonlinearFactorGraph graph_interp =
        interpolateFactorGraph<Pose2>(graph, estim, interp, sigma_wnoa, fixed_noise);
    // Run optimizer
    result_interp =
        LevenbergMarquardtOptimizer(graph_interp, initial, params).optimize();
    // save intermediate result with only estimated states
    saveResultToFile(result_interp, graph_interp, interp_raw_file, solve_slam);
    // Recover interpolated means using interpolator
    result = updateInterpValues<Pose2>(graph_interp, result_interp, estim,
                                       interp, sigma_wnoa);
  } else {
    result = LevenbergMarquardtOptimizer(graph, initial, params).optimize();
  }
  // Save results
  cout << "Optimizer has finished...saving results..." << endl;
  saveResultToFile(result, graph, output_file, solve_slam);
  saveResultToFile(gt, graph, "results/lost_gt.csv");

  return 0;
}
