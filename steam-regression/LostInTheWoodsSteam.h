#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "lgmath.hpp"
#include "steam.hpp"
#include "br2d_error_evaluator.h"

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

/** \brief Structure to store trajectory state variables */
struct TrajStateVar {
  Time time;
  se3::SE3StateVar::Ptr pose;
  vspace::VSpaceStateVar<6>::Ptr velocity;
};

/** \brief Function to save data to CSV */
int saveResultToFile(vector<TrajStateVar>& states, Covariance& cov_post,
                     const string& filename) {
  // open file, print header
  ofstream poses_file(filename);
  if (poses_file.is_open()) {
    poses_file << "x,y,theta,C11,C12,C13,C22,C23,C33\n";  // Header for Pose2
    // loop through states
    bool first_state = true;
    for (const auto& state : states) {
      // extract pose (NOTE: b : Body, a : Inertial)
      lgmath::se3::Transformation pose = state.pose->value();
      Eigen::Vector3d trans = pose.r_ba_ina();  //
      Eigen::Matrix3d C_ab = pose.C_ba().transpose();
      double theta = lgmath::so3::Rotation(C_ab)
                         .vec()[2];  // extract z component of vector
      Eigen::Matrix<double, 6, 6> cov;
      if (first_state) {
        cov = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
        first_state = false;
      } else {
        cov = cov_post.query(state.pose);
      }
      poses_file << trans[0] << "," << trans[1] << "," << theta << ","
                 << cov(0, 0) << "," << cov(0, 1) << "," << cov(0, 5) << ","
                 << cov(1, 1) << "," << cov(1, 5) << "," << cov(5, 5) << "\n";
    }
    poses_file.close();
    return 1;
  } else {
    cerr << "Error opening file" << endl;
    return 0;
  }
}
