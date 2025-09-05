#include <gtsam/geometry/BearingRange.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/nonlinear/ExpressionFactorGraph.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/expressions.h>
#include <gtsam/nonlinear/WNOAFactor.h>
#include <gtsam/nonlinear/WNOAInterpFactor.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
using namespace gtsam;

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
int saveResultToFile(Values& result, NonlinearFactorGraph& graph,
                     const string& filename, bool save_landmarks = false, std::shared_ptr<typename Interpolator<Pose2>::CovarianceMap> cov_map = nullptr) {
  
  cout << "Writing solve output to " << filename << endl;
  // Get marginals
  Marginals marginals(graph, result, Marginals::Factorization::QR);
  // open file, print header
  ofstream poses_file(filename);
  if (poses_file.is_open()) {
    poses_file
        << "key,x,y,theta,C11,C12,C13,C22,C23,C33\n";  // Header for Pose2
    // filter results for pose2
    for (const auto& [key, pose] : result.extract<Pose2>()) {
      // check if key is defined in covariance map before using the marginals
      Matrix cov;
      if (cov_map && cov_map->count(key)){
        cov = (*cov_map)[key];
      }else{
        cov = marginals.marginalCovariance(key);
      }
      poses_file << key << "," << pose.x() << "," << pose.y() << ","
                 << pose.theta() << "," << cov(0, 0) << "," << cov(0, 1) << ","
                 << cov(0, 2) << "," << cov(1, 1) << "," << cov(1, 2) << ","
                 << cov(2, 2) << "\n";
    }
    poses_file.close();
  } else {
    cerr << "Error opening file" << endl;
    return 0;
  }

  if(save_landmarks) {
    string filename_lm = filename;
    filename_lm.replace(filename_lm.find(".csv"), 4, "_landmarks.csv");
    // open file, print header
    ofstream landmarks_file(filename_lm);
    if (landmarks_file.is_open()) {
      landmarks_file << "key,x,y,C11,C12,C22\n";  // Header for Point2
      // filter results for Point2
      for (const auto& [key, point] : result.extract<Point2>()) {
        Matrix cov = marginals.marginalCovariance(key);
        landmarks_file << key << "," << point.x() << "," << point.y() << ","
                      << cov(0, 0) << "," << cov(0, 1) << ","
                      << cov(1, 1) << "\n";
      }
      landmarks_file.close();
    }
    else {
      cerr << "Error opening file" << endl;
      return 0;
    }
  }
  return 1;
}

typedef BearingRange<Pose2, Point2> BearingRange2;
typedef Expression<BearingRange2> BearingRange2_;

// Expression function for Range-Bearing to fixed Landmark factor
BearingRange2_ BearingRangeLandmarkPrediction(Key posekey,
                                              const Point2 landmark,
                                              const Pose2 T_vs) {
  // Define Expression for pose and landmark
  Pose2_ T_iv(posekey);
  Point2_ landmark_(landmark);
  // Compose transformation to get sensor frame
  Pose2_ T_is = compose(T_iv, Pose2_(T_vs));
  // Compute the bearing and range to the point
  return BearingRange2_(BearingRange2::Measure, T_is, landmark_);
}

// Expression function for Range-Bearing to non-fixed Landmark factor
BearingRange2_ BearingRangeLandmarkPredictionSLAM(Key posekey,
                                              const Key landmark,
                                              const Pose2 T_vs) {
  // Define Expression for pose and landmark
  Pose2_ T_iv(posekey);
  Point2_ landmark_(landmark);
  // Compose transformation to get sensor frame
  Pose2_ T_is = compose(T_iv, Pose2_(T_vs));
  // Compute the bearing and range to the point
  return BearingRange2_(BearingRange2::Measure, T_is, landmark_);
}
