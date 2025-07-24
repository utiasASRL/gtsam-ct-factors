#include "LostInTheWoodsExample.h"

int main(int argc, char* argv[]) {
  // input processing
  string output_file = "results/lost.csv";
  string config_file = "examples/Data/LostInTheWoods.yaml";
  if (argc > 1) {
    config_file = argv[1];
  }

  YAML::Node config = YAML::LoadFile(config_file);

  // load dataset
  string input_file = config["files"]["input"].as<string>();
  DatasetLoader data;
  string filename = findExampleDataFile(config["files"]["input"].as<string>());
  data.loadFromFile(filename);
  data.checkSizes();

  // switches for factors/init
  bool include_prior = config["flags"]["prior"].as<bool>();
  bool include_odom = config["flags"]["odom"].as<bool>();
  bool include_wnoa = config["flags"]["wnoa"].as<bool>();
  bool include_br_meas = config["flags"]["br"].as<bool>();
  bool gt_init = config["flags"]["gt_init"].as<bool>();
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
  Matrix3 Q_wnoa = sigma_wnoa.asDiagonal();

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
      // Define Keys
      Pose2_ curr(Symbol('x', i));
      Pose2_ prev(Symbol('x', i - 1));
      // define odometry measurement
      Pose2 odom(data.v[i - 1] * del_t, 0.0, data.om[i - 1] * del_t);
      // add factor to graph
      graph.addExpressionFactor(between(prev, curr), odom, odoNoise);
    }
  }

  // White-Noise-On-Acceleration Prior
  if (include_wnoa) {
    cout << "Adding WNOA factors" << endl;
    // Add WNOA Motion Factors between states
    for (int i = start + 1; i <= end; i++) {
      graph.add(WNOAMotionFactor<Pose2>(Symbol('x', i - 1), Symbol('v', i - 1),
                                        Symbol('x', i), Symbol('v', i), del_t,
                                        Q_wnoa));
    }
  }

  // BearingRange Measurements
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
        // Check if we have a valid measurement
        if ((data.range(i, j) > 0.0) && (abs(data.bearing(i, j)) > 0.0) &&
            (data.range(i, j) < r_max)) {
          // Get Bearing Range measurement
          BearingRange2 measurement(Rot2(data.bearing(i, j)), data.range(i, j));
          // Compute the bearing and range Prediction
          auto predict = BearingRangeLandmarkPrediction(xi, landmarks[j], T_vs);
          // Define Factor
          graph.addExpressionFactor(predict, measurement, measNoise);
        }
      }
    }
  }
  // Get ground truth solution
  Values gt;
  for (int i = start; i <= end; i++) {
    gt.insert(Symbol('x', i),
              Pose2(data.x_true[i], data.y_true[i], data.th_true[i]));
    if (include_wnoa) {
      gt.insert(Symbol('v', i), Vector3(data.v[i], 0.0, data.om[i]));
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

  // Run optimizer
  LevenbergMarquardtParams params;
  params.setVerbosityLM("SUMMARY");
  Values result =
      LevenbergMarquardtOptimizer(graph, initial, params).optimize();
  // Save results
  cout << "Optimizer has finished...saving results..." << endl;

  saveResultToFile(result, graph, output_file);
  saveResultToFile(gt, graph, "results/lost_gt.csv");

  return 0;
}
