#include <yaml-cpp/yaml.h>

#include <chrono>
#include <string>
#include <vector>

#include "LostInTheWoodsExample.h"
#include <gtsam/nonlinear/WNOAFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/WNOALevenbergMarquardtOptimizer.h>

using std::string;
using std::vector;

struct TimingParams {
  // File paths
  string input_file;
  string output_file;
  string gt_output_file;
  string interp_out;

  // Flags
  bool include_prior;
  bool include_odom;
  bool include_wnoa;
  bool include_br_meas;
  bool gt_init;
  bool solve_slam;

  // Interpolation
  uint interp_period;
  bool fixed_noise;

  // Parameters
  double r_max;
  double del_t;
  int start;
  int end;
  int n_runs;

  // Noise
  vector<double> sigma_prior_vec;
  vector<double> sigma_wnoa_vec;
  double sigma_y_odom;
  double mult_bearing;
  double mult_range;

  // Constructor to load from YAML node
  TimingParams(const YAML::Node& config) {
    input_file = config["files"]["input"].as<string>();
    output_file = config["files"]["output"].as<string>();
    gt_output_file = config["files"]["gt_out"].as<string>();
    interp_out = config["files"]["interp_out"].as<string>();

    include_prior = config["flags"]["prior"].as<bool>();
    include_odom = config["flags"]["odom"].as<bool>();
    include_wnoa = config["flags"]["wnoa"].as<bool>();
    include_br_meas = config["flags"]["br"].as<bool>();
    gt_init = config["flags"]["gt_init"].as<bool>();
    solve_slam = config["flags"]["solve_slam"].as<bool>();

    interp_period = config["interp"]["interp_period"].as<uint>();
    fixed_noise = config["interp"]["fixed_noise"].as<bool>();

    r_max = config["params"]["r_max"].as<double>();
    del_t = config["params"]["del_t"].as<double>();
    start = config["params"]["start"].as<int>();
    end = config["params"]["end"].as<int>();
    n_runs = config["params"]["n_runs"].as<int>();

    sigma_prior_vec = config["noise"]["prior"].as<vector<double>>();
    sigma_wnoa_vec = config["noise"]["wnoa"].as<vector<double>>();
    sigma_y_odom = config["noise"]["odom_y"].as<double>();
    mult_bearing = config["noise"]["bearing"].as<double>();
    mult_range = config["noise"]["range"].as<double>();
  }
};

int runLostInTheWoods(TimingParams& params) {
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
  uint interp_period = params.interp_period;
  string output_file_original = params.output_file;
  string output_file_interp = params.interp_out;

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
    if (include_wnoa) {
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
    if (include_wnoa) {
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
        if (include_wnoa) {
          initial.insert(Symbol('v', i), zero);
        }
      } else {
        Vector vel = Vector3(data.v[i - 1], 0.0, data.om[i - 1]);
        Vector3 vel_t = del_t * vel;
        Pose2 odom = Pose2::Expmap(vel_t);
        initial.insert(Symbol('x', i),
                       initial.at<Pose2>(Symbol('x', i - 1)).compose(odom));
        if (include_wnoa) {
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


  // set up optimizer parameters
  LevenbergMarquardtParams opt_params;
  opt_params.setVerbosityLM("SUMMARY");

  LevenbergMarquardtParams opt_params_silent;
  opt_params_silent.setVerbosityLM("SILENT");

  Values result_full, result_interp;

  // WARMUP runs
  //for(unsigned int i = 0; i < 20; i++)
  //{
  //  auto LM_warmup = LevenbergMarquardtOptimizer(graph, initial, opt_params_silent);
  //  LM_warmup.optimize();
  //}

  // set up stopwatch
  auto t_start = chrono::steady_clock::now();
  auto t_end = chrono::steady_clock::now();
  auto t_runtime =
      chrono::duration_cast<chrono::microseconds>(t_end - t_start).count();

  // Run optimizer for original graph
  std::cout << "Number of factors in original graph: " << graph.size() << std::endl;
  result_full = LevenbergMarquardtOptimizer(graph, initial, opt_params).optimize();

  

  t_start = chrono::steady_clock::now();
  for(unsigned int i = 0; i < params.n_runs; i++)
  {
    auto LM_opt = LevenbergMarquardtOptimizer(graph, initial, opt_params_silent);
    LM_opt.optimize();
  }
  t_end = chrono::steady_clock::now();
  t_runtime =
      chrono::duration_cast<chrono::microseconds>(t_end - t_start).count();
  cout << "Runtime for solving original graph: " << t_runtime / params.n_runs << " (micro-s)" << endl;
  

  // Run optimizer for wrapper graph
  // process states into estimated and interpolated
  set<StateData> interp;
  set<StateData> estim;
  set<StateData> all;
  for (size_t i = 0; i < all_states.size(); i++) {
    if (i == 0 || i == all_states.size() - 1 || i % interp_period == 0) {
      estim.insert(all_states[i]);
    } else {
      interp.insert(all_states[i]);
      // remove interpolated states from initial values
      initial.erase(all_states[i].pose);
      initial.erase(all_states[i].vel);
    }
    all.insert(all_states[i]);
  }

  t_start = chrono::steady_clock::now();
  for(unsigned int i = 0; i < params.n_runs; i++)
  {
    // Generate interpolated version of graph
    WNOAFactorGraph graph_interp = interpolateWNOAFactorGraph<Pose2>(
        graph, estim, interp, sigma_wnoa, params.fixed_noise);
  }
  t_end = chrono::steady_clock::now();
  t_runtime = chrono::duration_cast<chrono::microseconds>(t_end - t_start).count();
  cout << "Runtime for generating interpolated graph: " << t_runtime / params.n_runs << " (micro-s)" << endl;


  WNOAFactorGraph graph_interp = interpolateWNOAFactorGraph<Pose2>(
      graph, estim, interp, sigma_wnoa, params.fixed_noise);

  // Run optimizer

  std::cout << "Number of factors in interpolated graph: " << graph_interp.size() << std::endl;
  result_interp = WNOALevenbergMarquardtOptimizer<Pose2>(graph_interp, initial, opt_params).optimize();
  t_start = chrono::steady_clock::now();
  for(unsigned int i = 0; i < params.n_runs; i++)
  {
    auto LM_inter = WNOALevenbergMarquardtOptimizer<Pose2>(graph_interp, initial, opt_params_silent);
    LM_inter.optimize();
  }
  t_end = chrono::steady_clock::now();
  t_runtime = chrono::duration_cast<chrono::microseconds>(t_end - t_start).count();
  cout << "Runtime for solving wrapper graph: " << t_runtime / params.n_runs << " (micro-s)" << endl;

  // Recover interpolated means using interpolator
  std::shared_ptr<typename Interpolator<Pose2>::CovarianceMap> cov_map = std::make_shared<Interpolator<Pose2>::CovarianceMap>();


  // Define interpolator
  Interpolator<Pose2> interpolator(sigma_wnoa);

  t_start = chrono::steady_clock::now();
  for(unsigned int i = 0; i < params.n_runs; i++)
  {
  Values tmp = interpolator.interpolatePosesAndVelocities(
      graph_interp, result_interp, estim, all, nullptr);
  }
  t_end = chrono::steady_clock::now();
  t_runtime = chrono::duration_cast<chrono::microseconds>(t_end - t_start).count();
  cout << "Runtime for recovering interpolated means: " << t_runtime / params.n_runs << " (micro-s)" << endl;


  // get interpolated values
  Values result_restored = interpolator.interpolatePosesAndVelocities(
      graph_interp, result_interp, estim, all, cov_map);

  // if solve slam, grab landmarks from the result_interp
  if (solve_slam) {
    for (int j = 0; j < data.n_landmarks; j++) {
      Key landmark = Symbol('l', j);
      if (result_interp.exists(landmark)) {
        result_restored.insert(landmark, result_interp.at<Point2>(landmark));
      }
    }
  }

  result_interp = result_restored;

  // Save results
  cout << "Optimizer has finished...saving results..." << endl;
  saveResultToFile(result_full, graph, params.output_file, solve_slam, nullptr);
  saveResultToFile(gt, graph, params.gt_output_file);
  // Save results, interpolate covariances from graph at interp mean
  saveResultToFile(result_interp, graph, params.interp_out, solve_slam, cov_map);

  // Compute position RMSE for between resul_full and gt as well as result_interp and gt
  std::vector<Point2> errors_full, errors_interp, errors_interp_full, errors_interp_est_times;
  std::vector<Rot2> errors_full_rot, errors_interp_rot, errors_interp_full_rot;
  std::vector<double> nees_interp, nees_interp_est_times;
  for(int i = start; i <= end; i++) {
    Pose2 pose_full = result_full.at<Pose2>(Symbol('x', i));
    Pose2 pose_interp = result_interp.at<Pose2>(Symbol('x', i));
    Pose2 pose_gt = gt.at<Pose2>(Symbol('x', i));
    Point2 err_full = (pose_full.inverse().compose(pose_gt)).translation();
    Point2 err_interp = (pose_interp.inverse().compose(pose_gt)).translation();
    Point2 err_interp_full = (pose_interp.inverse().compose(pose_full)).translation();
    Rot2 err_full_rot = (pose_full.inverse().compose(pose_gt)).rotation();
    Rot2 err_interp_rot = (pose_interp.inverse().compose(pose_gt)).rotation();
    Rot2 err_interp_full_rot = (pose_interp.inverse().compose(pose_full)).rotation();
    errors_full.push_back(err_full);
    errors_interp.push_back(err_interp);
    errors_interp_full.push_back(err_interp_full);
    errors_full_rot.push_back(err_full_rot);
    errors_interp_rot.push_back(err_interp_rot);
    errors_interp_full_rot.push_back(err_interp_full_rot);



    Matrix2 cov_interp = cov_map->at(Symbol('x', i)).topLeftCorner<2,2>();
    double nees_i = err_interp.transpose() * cov_interp.inverse() * err_interp;
    nees_interp.push_back(nees_i);

    if (i == start || i == end || (i - start) % interp_period == 0) {
      errors_interp_est_times.push_back(err_interp);
      nees_interp_est_times.push_back(nees_i);
    }

  }

  // Compute RMSE values from vectors
  const size_t n = errors_full.size();
  double sum_sq_full = 0.0, sum_sq_interp = 0.0, sum_sq_interp_full = 0.0;
  double sum_sq_full_x = 0.0, sum_sq_full_y = 0.0;
  double sum_sq_interp_x = 0.0, sum_sq_interp_y = 0.0;
  double sum_sq_interp_full_x = 0.0, sum_sq_interp_full_y = 0.0;
  double sum_sq_full_rot = 0.0, sum_sq_interp_rot = 0.0, sum_sq_interp_full_rot = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double fx = errors_full[i].x(), fy = errors_full[i].y();
    double ix = errors_interp[i].x(), iy = errors_interp[i].y();
    double ix_full = errors_interp_full[i].x(), iy_full = errors_interp_full[i].y();
    sum_sq_full += fx * fx + fy * fy;
    sum_sq_interp += ix * ix + iy * iy;
    sum_sq_interp_full += ix_full * ix_full + iy_full * iy_full;
    sum_sq_full_x += fx * fx;
    sum_sq_full_y += fy * fy;
    sum_sq_interp_x += ix * ix;
    sum_sq_interp_y += iy * iy;
    sum_sq_interp_full_x += ix_full * ix_full;
    sum_sq_interp_full_y += iy_full * iy_full;
    double ftheta = errors_full_rot[i].degrees();
    double itheta = errors_interp_rot[i].degrees();
    double itheta_full = errors_interp_full_rot[i].degrees();
    sum_sq_full_rot += ftheta * ftheta;
    sum_sq_interp_rot += itheta * itheta;
    sum_sq_interp_full_rot += itheta_full * itheta_full;

  }

  //Compute RMSE at estimation times
  const size_t n_est = errors_interp_est_times.size();
  double sum_sq_interp_est = 0.0;
  double sum_sq_interp_x_est = 0.0, sum_sq_interp_y_est = 0.0;
  for( size_t i = 0; i < n_est; ++i) {
    double ix = errors_interp_est_times[i].x();
    double iy = errors_interp_est_times[i].y();
    sum_sq_interp_est += ix * ix + iy * iy;
    sum_sq_interp_x_est += ix * ix;
    sum_sq_interp_y_est += iy * iy;
  }

  double rmse_full = std::sqrt(sum_sq_full / n);
  double rmse_interp = std::sqrt(sum_sq_interp / n);
  double rmse_interp_full = std::sqrt(sum_sq_interp_full / n);
  double rmse_full_x = std::sqrt(sum_sq_full_x / n);
  double rmse_full_y = std::sqrt(sum_sq_full_y / n);
  double rmse_interp_x = std::sqrt(sum_sq_interp_x / n);
  double rmse_interp_y = std::sqrt(sum_sq_interp_y / n);
  double rmse_interp_full_x = std::sqrt(sum_sq_interp_full_x / n);
  double rmse_interp_full_y = std::sqrt(sum_sq_interp_full_y / n);
  double rmse_full_rot = std::sqrt(sum_sq_full_rot / n);
  double rmse_interp_rot = std::sqrt(sum_sq_interp_rot / n);
  double rmse_interp_full_rot = std::sqrt(sum_sq_interp_full_rot / n);

  double rmse_interp_est = std::sqrt(sum_sq_interp_est / n_est);
  double rmse_interp_x_est = std::sqrt(sum_sq_interp_x_est / n_est);
  double rmse_interp_y_est = std::sqrt(sum_sq_interp_y_est / n_est);

  double nees_interp_mean =  std::accumulate(nees_interp.begin(), nees_interp.end(), 0.0) / nees_interp.size();
  double nees_interp_est_times_mean =  std::accumulate(nees_interp_est_times.begin(), nees_interp_est_times.end(), 0.0) / nees_interp_est_times.size();

  // Print with higher precision
  cout << std::fixed << std::setprecision(8);
  cout << "--------------------------------------------------------------" << endl;
  cout << "RMSE (full)           : " << rmse_full << " m" << endl;
  //cout << "RMSE rot (full)       : " << rmse_full_rot << " deg" << endl;
  cout << "--------------------------------------------------------------" << endl;
  cout << "RMSE (interp)         : " << rmse_interp << " m" << endl;
  //cout << "RMSE rot (interp)     : " << rmse_interp_rot << " deg" << endl;
  cout << "NEES (interp)         : " << nees_interp_mean << endl;
  cout << "--------------------------------------------------------------" << endl;
  cout << "RMSE (interp, est ts) : " << rmse_interp_est << " m" << endl;
  cout << "NEES (interp, est ts) : " << nees_interp_est_times_mean << endl;
  cout << "--------------------------------------------------------------" << endl;
  //cout << "RMSE (interp-full)    : " << rmse_interp_full << " m" << endl;
  //cout << "RMSE rot (interp-full): " << rmse_interp_full_rot << " deg" << endl;
  //cout << "--------------------------------------------------------------" << endl;
  // restore default formatting (optional)
  cout << std::defaultfloat;


  return 0;
}

int main(int argc, char* argv[]) {
  // Get configuration data
  string config_file = "LostInTheWoods/timing_params.yaml";
  if (argc > 1) {
    config_file = argv[1];
  }
  YAML::Node config = YAML::LoadFile(config_file);

  // Use parameter struct to load all parameters
  TimingParams params(config);

  runLostInTheWoods(params);
}
