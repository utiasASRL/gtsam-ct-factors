#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "lgmath.hpp"
#include "steam.hpp"

using namespace std;
using namespace steam;
using namespace steam::traj;

class DatasetLoader {
 private:
  // Store variable name to string (you can later convert to float, etc.)
  unordered_map<string, string> variables;

 public:
  // singlve var definitions
  double b_var;
  double d;
  double om_var;
  double r_var;
  double v_var;

  // vectors
  Eigen::VectorXd om;
  Eigen::VectorXd v;
  Eigen::VectorXd t;
  Eigen::VectorXd th_true;
  Eigen::VectorXd x_true;
  Eigen::VectorXd y_true;

  // matrices
  Eigen::MatrixXd landmarks;
  Eigen::MatrixXd bearing;
  Eigen::MatrixXd range;

  // sizes
  int size = 0;
  int n_landmarks = 0;

 public:
  void loadFromFile(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
      std::cerr << "Could not open file: " << filename << "\n";
      return;
    }

    std::string line, varName, varValue;
    bool readingValue = false;

    while (std::getline(infile, line)) {
      if (line.rfind("Variable: ", 0) == 0) {
        if (!varName.empty()) {
          variables[varName] = varValue;
          varValue.clear();
        }

        varName = line.substr(10);
        readingValue = true;
      } else if (readingValue) {
        if (!varValue.empty()) varValue += " ";
        varValue += line;
      }
    }

    if (!varName.empty()) {
      variables[varName] = varValue;
    }

    infile.close();

    // Parse the variables after loading
    parseNumerics();

    cout << "Loaded " << variables.size() << " variables from " << filename
         << "\n";
  }

  // Parse a matrix from a MATLAB-style string into an Eigen::MatrixXd
  Eigen::MatrixXd parseEigenMatrix(const std::string& raw) {
    std::string s = raw;

    if (!s.empty() && s.front() == '[') s.erase(0, 1);
    if (!s.empty() && s.back() == ']') s.pop_back();

    std::vector<std::vector<double>> rows;
    std::stringstream ss(s);
    std::string rowStr;

    while (std::getline(ss, rowStr, ';')) {
      std::istringstream rowStream(rowStr);
      std::vector<double> row;
      double val;
      while (rowStream >> val) {
        row.push_back(val);
      }
      if (!row.empty()) rows.push_back(row);
    }

    if (rows.empty()) return Eigen::MatrixXd();

    size_t numRows = rows.size();
    size_t numCols = rows[0].size();
    Eigen::MatrixXd mat(numRows, numCols);

    for (size_t i = 0; i < numRows; ++i) {
      if (rows[i].size() != numCols) {
        throw std::runtime_error("Inconsistent row sizes in matrix.");
      }
      for (size_t j = 0; j < numCols; ++j) {
        mat(i, j) = rows[i][j];
      }
    }

    return mat;
  }

  Eigen::VectorXd flattenIfVector(const Eigen::MatrixXd& mat) {
    if (mat.cols() == 1) {
      // Already a column vector
      return mat.col(0);
    } else if (mat.rows() == 1) {
      // Row vector → transpose to column
      return mat.row(0).transpose();
    } else {
      throw std::runtime_error(
          "Matrix is not a vector (1 row or 1 column required)");
    }
  }

  void parseNumerics() {
    b_var = stod(variables["b_var"]);
    d = stod(variables["d"]);
    om_var = stod(variables["om_var"]);
    r_var = stod(variables["r_var"]);
    v_var = stod(variables["v_var"]);
    om = flattenIfVector(parseEigenMatrix(variables["om"]));
    v = flattenIfVector(parseEigenMatrix(variables["v"]));
    t = flattenIfVector(parseEigenMatrix(variables["t"]));
    th_true = flattenIfVector(parseEigenMatrix(variables["th_true"]));
    x_true = flattenIfVector(parseEigenMatrix(variables["x_true"]));
    y_true = flattenIfVector(parseEigenMatrix(variables["y_true"]));
    bearing = parseEigenMatrix(variables["b"]);
    range = parseEigenMatrix(variables["r"]);
    landmarks = parseEigenMatrix(variables["l"]);
    size = y_true.size();
    n_landmarks = landmarks.rows();
  }

  void checkSizes() {
    cout << "b_var: " << b_var << "\n";
    cout << "d: " << d << "\n";
    cout << "om_var: " << om_var << "\n";
    cout << "r_var: " << r_var << "\n";
    cout << "v_var: " << v_var << "\n";
    cout << "om shape: " << om.rows() << " x " << om.cols() << "\n";
    cout << "t shape: " << t.rows() << " x " << t.cols() << "\n";
    cout << "th_true shape: " << th_true.rows() << " x " << th_true.cols()
         << "\n";
    cout << "x_true shape: " << x_true.rows() << " x " << x_true.cols() << "\n";
    cout << "y_true shape: " << y_true.rows() << " x " << y_true.cols() << "\n";
    cout << "bearing shape: " << bearing.rows() << " x " << bearing.cols()
         << "\n";
    cout << "range shape: " << range.rows() << " x " << range.cols() << "\n";
    cout << "landmarks shape: " << landmarks.rows() << " x " << landmarks.cols()
         << "\n";
    cout << "traj len: " << size << endl;
  }
};

// --- Save Poses to CSV ---
// int saveResultToFile(Values& result, NonlinearFactorGraph& graph,
//                      const string& filename) {
//   // Get marginals
//   Marginals marginals(graph, result);

//   // open file, print header
//   ofstream poses_file(filename);
//   if (poses_file.is_open()) {
//     poses_file
//         << "key,x,y,theta,C11,C12,C13,C22,C23,C33\n";  // Header for Pose2
//     // filter results for pose2
//     for (const auto& [key, pose] : result.extract<Pose2>()) {
//       Matrix cov = marginals.marginalCovariance(key);
//       poses_file << key << "," << pose.x() << "," << pose.y() << ","
//                  << pose.theta() << "," << cov(0, 0) << "," << cov(0, 1) <<
//                  ","
//                  << cov(0, 2) << "," << cov(1, 1) << "," << cov(1, 2) << ","
//                  << cov(2, 2) << "\n";
//     }
//     poses_file.close();
//     return 1;
//   } else {
//     cerr << "Error opening file" << endl;
//     return 0;
//   }
// }

/** \brief Structure to store trajectory state variables */
struct TrajStateVar {
  Time time;
  se3::SE3StateVar::Ptr pose;
  vspace::VSpaceStateVar<6>::Ptr velocity;
};

int main(int argc, char* argv[]) {
  // Get configuration data
  string config_file = "steam-regression/LostInTheWoods.yaml";
  if (argc > 1) {
    config_file = argv[1];
  }
  YAML::Node config = YAML::LoadFile(config_file);

  // Load Files
  string input_file = config["files"]["input"].as<string>();
  string output_file = config["files"]["output"].as<string>();
  string gt_output_file = config["files"]["gt_out"].as<string>();
  // Load dataset
  DatasetLoader data;
  data.loadFromFile(input_file);
  data.checkSizes();
  // switches for factors/init
  bool include_prior = config["flags"]["prior"].as<bool>();
  bool include_odom = config["flags"]["odom"].as<bool>();
  bool include_br_meas = config["flags"]["br"].as<bool>();
  bool gt_init = config["flags"]["gt_init"].as<bool>();
  // Get inputs from param file
  double r_max = config["params"]["r_max"].as<double>();
  double del_t = config["params"]["del_t"].as<double>();
  int start = config["params"]["start"].as<int>();
  int end = config["params"]["end"].as<int>();
  // Get noise model parameters
  auto sigma_prior =
      Eigen::Vector3d(config["noise"]["prior"].as<vector<double>>().data());
  auto sigma_wnoa =
      Eigen::Vector3d(config["noise"]["wnoa"].as<vector<double>>().data());
  double sigma_y_odom = config["noise"]["odom_y"].as<double>();
  double mult_bearing = config["noise"]["bearing"].as<double>();
  double mult_range = config["noise"]["range"].as<double>();
  auto sigma_odom =
      Eigen::Vector3d(sqrt(data.v_var), sigma_y_odom, sqrt(data.om_var)) *
      del_t;
  auto sigma_br = Eigen::Vector2d(sqrt(mult_bearing * data.b_var),
                                  sqrt(mult_range * data.r_var));

  // Starting point
  Eigen::Matrix<double, 6, 1> startPoseVec, startVel;
  startPoseVec << data.x_true[start], data.y_true[start], 0.0, 0.0, 0.0,
      data.th_true[start];
  lgmath::se3::Transformation startPose(startPoseVec);
  startVel << data.v[start], 0.0, 0.0, 0.0, 0.0, data.om[start];

  ///
  /// Setup States
  ///

  // States
  vector<TrajStateVar> states;
  // Initialization
  for (int i = start; i <= end; i++) {
    // current velocity for odometry
    Eigen::Matrix<double, 6, 1> vel(data.v[i], 0.0, 0.0, 0.0, 0.0,
                                      data.om[i]);
    if (i == start || gt_init) { // Use gt for initial pose, or if initing from gt
      // get pose and vel
      Eigen::Matrix<double, 6, 1> poseVec(data.x_true[i], data.y_true[i], 0.0,
                                          0.0, 0.0, data.th_true[i]);
      lgmath::se3::Transformation pose(poseVec);
      // push to vector of states
      TrajStateVar temp;
      temp.time = Time(i * del_t);
      temp.pose = se3::SE3StateVar::MakeShared(pose);
      temp.velocity = vspace::VSpaceStateVar<6>::MakeShared(vel);
      states.emplace_back(temp);
    } else {  // otherwise roll out odometry
      // Get relative pose
      TrajStateVar prev = states.back();
      
      lgmath::se3::Transformation T_rel(states.back().velocity->value() * del_t);
      // push to vector of states
      TrajStateVar temp;
      temp.time = Time(i * del_t);
      temp.pose = se3::SE3StateVar::MakeShared(T_rel * states.back().pose->value());
      temp.velocity = vspace::VSpaceStateVar<6>::MakeShared(vel);
      states.emplace_back(temp);
    } 

  }

  //define optimization
  OptimizationProblem problem;
  // Add state variables
  for (const auto& state : states) {
    problem.addStateVariable(state.pose);
    problem.addStateVariable(state.velocity);
  }

  // Setup WNOA Prior
  Eigen::Matrix<double, 6,1> Qc_diag(sigma_wnoa[0],sigma_wnoa[1],0.0,0.0,0.0, sigma_wnoa[2]);
  traj::const_vel::Interface traj(Qc_diag);
  for (const auto& state : states)
    traj.add(state.time, state.pose, state.velocity);
  traj.addPriorCostTerms(problem);

  

} 

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

// Run optimizer
LevenbergMarquardtParams params;
params.setVerbosityLM("SUMMARY");
Values result = LevenbergMarquardtOptimizer(graph, initial, params).optimize();
// Save results
cout << "Optimizer has finished...saving results..." << endl;

saveResultToFile(result, graph, output_file);
saveResultToFile(gt, graph, "results/lost_gt.csv");

return 0;
}