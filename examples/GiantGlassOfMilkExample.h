// C++ includes
#include <iostream>

// Eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

// GTSAM
#include <gtsam/geometry/Point1.h>
#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/WNOAFactor.h>
#include <gtsam/nonlinear/WNOAInterpolator.h>
#include <gtsam/nonlinear/expressions.h>
#include <gtsam/slam/dataset.h>
#include <yaml-cpp/yaml.h>

using namespace gtsam;

// Load CSV files to Eigen Matrix
// From:
// https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix
template <typename M>
M load_csv(const std::string& path, bool skip_header = false) {
  std::ifstream indata;
  indata.open(path);
  std::string line;
  std::vector<double> values;
  uint rows = 0;
  if (skip_header && std::getline(indata, line)) {
    // Skip the header line
  }
  while (std::getline(indata, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      // Check if cell is empty and if so assign nan
      try {
        std::stod(cell);
      } catch (const std::invalid_argument&) {
        values.push_back(nan(""));
        continue;
      }
      values.push_back(std::stod(cell));
    }

    ++rows;
  }
  return Eigen::Map<
      const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime,
                          M::ColsAtCompileTime, Eigen::RowMajor>>(
      values.data(), rows, values.size() / rows);
}

// Save Eigen Matrix to CSV file
template <typename M>
void save_csv(const std::string& path, const M& matrix) {
  std::ofstream outdata(path);
  if (!outdata.is_open()) {
    throw std::runtime_error("Could not open file for writing: " + path);
  }
  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      outdata << matrix(i, j);
      if (j != matrix.cols() - 1) outdata << ",";
    }
    outdata << "\n";
  }
  outdata.close();
}
