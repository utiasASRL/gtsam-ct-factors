#include "LostInTheWoodsExample.h"

int main(int argc, char* argv[]) {
  // input processing
  string input_file = "LostInTheWoods.txt";
  string output_file = "../results/lost.csv";
  int start = 0;    // trajectory start
  int end = 12600;  // trajectory end
  if (argc > 1) {
    start = atoi(argv[1]);
  }
  if (argc > 1) {
    end = atoi(argv[2]);
  }
  if (argc > 3) {
    output_file = argv[3];
  }
  if (argc > 4) {
    input_file = argv[4];
  }

  // load dataset
  DatasetLoader data;
  string filename = findExampleDataFile(input_file);
  data.loadFromFile(filename);
  data.checkSizes();

  // switches for factors/init
  bool include_prior = false;
  bool include_odom = false;
  bool include_wnoa = true;
  bool include_br_meas = true;
  bool gt_init = true;

  // Create a factor graph
  ExpressionFactorGraph graph;

  // Get Noise Models
  double del_t = 0.1;
  Vector priorSigmas = Vector3(0.5, 0.5, 0.5);
  Vector odoSigmas = Vector3(sqrt(data.v_var), 1e-1, sqrt(data.om_var)) * del_t;
  Vector BRSigmas = Vector2(sqrt(data.b_var), sqrt(data.r_var));
  auto priorNoise = noiseModel::Diagonal::Sigmas(priorSigmas);  // prior
  auto odoNoise = noiseModel::Diagonal::Sigmas(odoSigmas);      // odometry
  auto measNoise =
      noiseModel::Diagonal::Sigmas(BRSigmas);  // range-bearing noise
  // WNOA Factor
  Vector Q_wnoa = 0.1 * Vector3(1.0, 1.0, 0.1);
  // Starting point
  Pose2 startPose(data.x_true[start], data.y_true[start], data.th_true[start]);

  // Prior
  if (include_prior) {
    graph.add(PriorFactor<Pose2>(Symbol('x', start), startPose, priorNoise));
  }
  // Odometry factors
  if (include_odom) {
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
    for (int i = start + 1; i <= end; i++) {
      graph.add(WNOAMotionFactor<Pose2>(Symbol('x', i - 1), Symbol('v', i - 1),
                                        Symbol('x', i), Symbol('v', i), del_t,
                                        Q_wnoa));
    }
  }

  // BearingRange Measurements
  if (include_br_meas) {
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
        if ((data.range(i, j) > 0.0) && (abs(data.bearing(i, j)) > 0.0)) {
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

  // Initialization
  Values initial;
  for (int i = start; i <= end; i++) {
    if (gt_init) {
      // Initialize at ground truth
      initial.insert(Symbol('x', i),
                     Pose2(data.x_true[i], data.y_true[i], data.th_true[i]));
      initial.insert(Symbol('v', i), Vector3(data.v[i], 0.0, data.om[i]));
    } else {
      // Rollout odometry
      if (i == start) {
        initial.insert(Symbol('x', i), startPose);
        Vector3 zero = Vector3::Zero();
        initial.insert(Symbol('v', i), zero);
      } else {
        Vector vel = Vector3(data.v[i - 1], 0.0, data.om[i - 1]);
        Vector3 vel_t = del_t * vel;
        Pose2 odom = Pose2::Expmap(vel_t);
        initial.insert(Symbol('x', i),
                       initial.at<Pose2>(Symbol('x', i - 1)).compose(odom));
        initial.insert(Symbol('v', i), vel);
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

  return 0;
}
