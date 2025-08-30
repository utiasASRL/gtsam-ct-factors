#include <yaml-cpp/yaml.h>

#include <chrono>
#include <string>
#include <vector>

#include "LostInTheWoodsExample.h"

using std::string;
using std::vector;

struct LostInTheWoodsParams {
  // File paths
  string input_file;
  string output_file;
  string gt_output_file;
  string interp_raw_file;
  string interp_out;
  string interp_graph_out;

  // Flags
  bool include_prior;
  bool include_odom;
  bool include_wnoa;
  bool include_br_meas;
  bool gt_init;
  bool solve_slam;

  // Interpolation
  bool interp_enable;
  uint interp_period;
  bool fixed_noise;

  // Parameters
  double r_max;
  double del_t;
  int start;
  int end;

  // Noise
  vector<double> sigma_prior_vec;
  vector<double> sigma_wnoa_vec;
  double sigma_y_odom;
  double mult_bearing;
  double mult_range;

  // Constructor to load from YAML node
  LostInTheWoodsParams(const YAML::Node& config) {
    input_file = config["files"]["input"].as<string>();
    output_file = config["files"]["output"].as<string>();
    gt_output_file = config["files"]["gt_out"].as<string>();
    interp_raw_file = config["files"]["interp_raw_file"].as<string>();
    interp_out = config["files"]["interp_out"].as<string>();
    interp_graph_out = config["files"]["interp_graph_out"].as<string>();

    include_prior = config["flags"]["prior"].as<bool>();
    include_odom = config["flags"]["odom"].as<bool>();
    include_wnoa = config["flags"]["wnoa"].as<bool>();
    include_br_meas = config["flags"]["br"].as<bool>();
    gt_init = config["flags"]["gt_init"].as<bool>();
    solve_slam = config["flags"]["solve_slam"].as<bool>();

    interp_enable = config["interp"]["enable"].as<bool>();
    interp_period = config["interp"]["interp_period"].as<uint>();
    fixed_noise = config["interp"]["fixed_noise"].as<bool>();

    r_max = config["params"]["r_max"].as<double>();
    del_t = config["params"]["del_t"].as<double>();
    start = config["params"]["start"].as<int>();
    end = config["params"]["end"].as<int>();

    sigma_prior_vec = config["noise"]["prior"].as<vector<double>>();
    sigma_wnoa_vec = config["noise"]["wnoa"].as<vector<double>>();
    sigma_y_odom = config["noise"]["odom_y"].as<double>();
    mult_bearing = config["noise"]["bearing"].as<double>();
    mult_range = config["noise"]["range"].as<double>();
  }
};

int runLostInTheWoods(LostInTheWoodsParams& params) {
  // Load Files
  DatasetLoader data;
  data.loadFromFile(params.input_file);
  // data.checkSizes();

  // switches for factors/init
  bool include_prior = params.include_prior;
  bool include_odom = params.include_odom;
  bool include_wnoa = params.include_wnoa;
  bool include_br_meas = params.include_br_meas;
  bool gt_init = params.gt_init;
  bool solve_slam = params.solve_slam;

  // interpolation
  bool interp_enable = params.interp_enable;
  uint interp_period = params.interp_period;
  string output_file = params.output_file;
  if (interp_enable) {
    output_file = params.interp_out;
  } else {
    output_file = params.output_file;
  }

  // Get inputs from param struct
  double r_max = params.r_max;
  double del_t = params.del_t;
  int start = params.start;
  int end = params.end;

  // Get noise model parameters
  Vector sigma_prior = Vector3(params.sigma_prior_vec.data());
  Vector sigma_wnoa = Vector3(params.sigma_wnoa_vec.data());
  double sigma_y_odom = params.sigma_y_odom;
  double mult_bearing = params.mult_bearing;
  double mult_range = params.mult_range;

  // Generate noise models
  Vector sigma_odom =
      Vector3(sqrt(data.v_var), sigma_y_odom, sqrt(data.om_var)) * del_t;
  Vector sigma_br =
      Vector2(sqrt(mult_bearing * data.b_var), sqrt(mult_range * data.r_var));
  auto priorNoise = noiseModel::Diagonal::Sigmas(sigma_prior);  // prior
  auto odoNoise = noiseModel::Diagonal::Sigmas(sigma_odom);     // odometry
  auto measNoise =
      noiseModel::Diagonal::Sigmas(sigma_br);  // range-bearing noise
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
    for (int i = start + 1; i <= end; i++) {
      graph.add(WNOAMotionFactor<Pose2>(Symbol('x', i - 1), Symbol('v', i - 1),
                                        Symbol('x', i), Symbol('v', i), del_t,
                                        sigma_wnoa));
    }
  }

  // BearingRange Measurements
  // Create a list of 18 booleans that track which landmarks have been observed
  // at least once
  vector<bool> landmark_observed(data.n_landmarks, false);
  if (include_br_meas) {
    cout << "Adding bearing range measurement factors" << endl;
    cout << "Max Range:  " << r_max << " (m)" << endl;

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
  if (solve_slam) {
    for (int j = 0; j < data.n_landmarks; j++) {
      gt.insert(Symbol('l', j),
                Point2(data.landmarks(j, 0), data.landmarks(j, 1)));
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
  if (solve_slam && !gt_init) {
    // Initialize landmarks at zero
    for (int j = 0; j < data.n_landmarks; j++) {
      // Only add keys for landmarks that have been observed
      if (landmark_observed[j]) {
        initial.insert(Symbol('l', j), Point2(0, 0));
      }
    }
  }

  // set up optimizer
  LevenbergMarquardtParams opt_params;
  opt_params.setVerbosityLM("SUMMARY");

  // set up stopwatch
  auto t_start = chrono::high_resolution_clock::now();
  auto t_end = chrono::high_resolution_clock::now();

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
    NonlinearFactorGraph graph_interp = interpolateFactorGraph<Pose2>(
        graph, estim, interp, sigma_wnoa, params.fixed_noise);
    // Run optimizer
    t_start = chrono::high_resolution_clock::now();
    result_interp =
        LevenbergMarquardtOptimizer(graph_interp, initial, opt_params)
            .optimize();
    t_end = chrono::high_resolution_clock::now();
    auto t_runtime =
        chrono::duration_cast<chrono::microseconds>(t_end - t_start).count();
    cout << "Runtime for solve: " << t_runtime << " (micro-s)" << endl;

    // Recover interpolated means using interpolator
    std::shared_ptr<typename Interpolator<Pose2>::CovarianceMap> cov_map;
    result = updateInterpValues<Pose2>(graph_interp, result_interp, estim,
                                       interp, sigma_wnoa, cov_map);
    // Save results
    cout << "Optimizer has finished...saving results..." << endl;
    // save only estimated states
    saveResultToFile(result_interp, graph_interp, params.interp_raw_file,
                     solve_slam);
    // Save results, interpolate covariances using funciton
    saveResultToFile(result, graph, params.interp_out, solve_slam, cov_map);
    // Save results, interpolate covariances from graph at interp mean
    saveResultToFile(result, graph, params.interp_graph_out, solve_slam);
    saveResultToFile(gt, graph, params.gt_output_file);
  } else {
    t_start = chrono::high_resolution_clock::now();
    result = LevenbergMarquardtOptimizer(graph, initial, opt_params).optimize();
    t_end = chrono::high_resolution_clock::now();
    auto t_runtime =
        chrono::duration_cast<chrono::microseconds>(t_end - t_start).count();
    cout << "Runtime for solve: " << t_runtime << " (micro-s)" << endl;
    // Save results
    cout << "Optimizer has finished...saving results..." << endl;
    saveResultToFile(result, graph, params.output_file, solve_slam);
    saveResultToFile(gt, graph, params.gt_output_file);
  }

  return 0;
}

int main(int argc, char* argv[]) {
  // Get configuration data
  string config_file = "LostInTheWoods/default_params.yaml";
  if (argc > 1) {
    config_file = argv[1];
  }
  YAML::Node config = YAML::LoadFile(config_file);

  // Use parameter struct to load all parameters
  LostInTheWoodsParams params(config);

  runLostInTheWoods(params);
}
