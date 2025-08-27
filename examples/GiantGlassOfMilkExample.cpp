#include "GiantGlassOfMilkExample.h"

int main() {

  // Get configuration data
  string config_file = "../examples/Data/GiantGlassOfMilk.yaml";
  YAML::Node config = YAML::LoadFile(config_file);

  // Load Files
  string input_file = config["files"]["input"].as<string>();
  string output_file = config["files"]["output"].as<string>();

  // input processing 
  bool use_interpolation = config["params"]["interp"].as<bool>();

  
  string filename = findExampleDataFile(input_file);

  // Loading data as: t, x, v, x_real
  Matrix data = load_csv<Eigen::MatrixXd>(filename, false);

  Vector times = data.col(0);
  Vector x_meas = data.col(1);
  Vector v_meas = data.col(2);
  Vector x_real = data.col(3);

  // Defining timing variables
  double dt = 0.1;
  double dt_meas = config["params"]["dt_meas"].as<double>();
  double dt_state = dt;
  if (use_interpolation) {
    dt_state = dt_meas;
  }

  //Define noise variables
  double Qc = 0.01;
  Vector1 Qc_mat;
  Qc_mat << Qc;
  auto positionNoise = noiseModel::Isotropic::Sigma(1,std::sqrt(3.6692e-03));
  auto velocityNoise = noiseModel::Isotropic::Sigma(1,std::sqrt(0.0023));

  // Initializing graph and values
  NonlinearFactorGraph graph;
  Values initialEstimate;


  Interpolator<Point1>::StateDataSet mainSolveStateSet;
  Interpolator<Point1>::StateDataSet interpolateStateSet;
  std::shared_ptr<Interpolator<Point1>::CovarianceMap> covarianceMap = std::make_shared<Interpolator<Point1>::CovarianceMap>();

  // Defining shorthand for symbols
  using symbol_shorthand::X; // state
  using symbol_shorthand::V; // velocity
  using symbol_shorthand::Z; // interpolted state
  using symbol_shorthand::D; // interpolated velocity


  // Add initial guess for first state (zero)
  initialEstimate.insert(X(0),Point1(0.0));
  initialEstimate.insert(V(0),Vector1(0.0));
  mainSolveStateSet.insert(StateData(X(0), V(0), times(0)));

  // Run through all states and add WNOA prior factors between neighbouring states
  unsigned int num_states = static_cast<unsigned int>(std::round(times.tail<1>()(0) / dt_state)) + 1;
  for(unsigned int i = 0; i < num_states - 1; i++)
  {
      graph.add(WNOAMotionFactor<Point1>(X(i),V(i),X(i+1),V(i+1),dt_state,Qc_mat));

      // Add initial guess for next state (zero)
      initialEstimate.insert(X(i+1),Point1(0.0));
      initialEstimate.insert(V(i+1),Vector1(0.0));
      mainSolveStateSet.insert(StateData(X(i+1), V(i+1), (i+1)*dt_state));
  }

  
  // Add keys for the interpolation
  for(unsigned int i = 0; i < times.size(); i++) {
      interpolateStateSet.insert(StateData(Z(i), D(i), times(i)));
  }

  // Add measurement factors for both position and velocity
  int skip = dt_meas/dt; // Skip factor, depending on the measurement frequency

  // times_est, x_est, x_std, v_est, v_std, x_real, times_meas, x_meas, v_meas
  Matrix result_matrix = Matrix::Zero(times.size(), 9);


  for(unsigned int i = 0; i < times.size(); i = i + skip)
  {
    // Save used measurements for plotting later
    result_matrix(i/skip, 6) = times(i);
    result_matrix(i/skip, 7) = x_meas(i);
    result_matrix(i/skip, 8) = v_meas(i);

    Point1 position_measurement(x_meas(i));
    Vector1 velocity_measurement(v_meas(i));

    if(use_interpolation) {
      graph.add(PriorFactor<Point1>(X(i/skip),position_measurement, positionNoise));
      graph.add(PriorFactor<Vector1>(V(i/skip),velocity_measurement, velocityNoise));
    }
    else {
      graph.add(PriorFactor<Point1>(X(i),position_measurement, positionNoise));
      graph.add(PriorFactor<Vector1>(V(i),velocity_measurement, velocityNoise));
    }

  }

  // Optimize
  std::cout << "Optimizing..." << std::endl;
  LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
  Values result = optimizer.optimize();
  Marginals marginals(graph, result);


  if (use_interpolation) {
    // interpolate states at requested times
    std::cout << "Querying..." << std::endl;
    Interpolator<Point1> interpolator(Qc_mat);
    Values results_interpolated = interpolator.interpolatePosesAndVelocities(graph, result, mainSolveStateSet, interpolateStateSet, covarianceMap);

    std::cout << "Saving values..." << std::endl;
    for(unsigned int i = 0; i < times.size(); i++)
    {
      // Time
      result_matrix(i, 0) = times(i);

      // X estimate
      result_matrix(i, 1) = results_interpolated.at<Point1>(Z(i))(0);
      // Get covariance of z from covarianceMap
      result_matrix(i, 2) = std::sqrt(covarianceMap->at(Z(i))(0,0));



      // V estimate
      result_matrix(i, 3) = results_interpolated.at<Vector1>(D(i))(0);
      result_matrix(i, 4) = std::sqrt(covarianceMap->at(D(i))(0,0));

      // X real
      result_matrix(i, 5) = x_real(i);
    }
  }
  else
  {

    std::cout << "Saving values..." << std::endl;
    for(unsigned int i = 0; i < times.size(); i++)
    {
      // Time
      result_matrix(i, 0) = times(i);

      // X estimate
      result_matrix(i, 1) = result.at<Point1>(X(i))(0);
      result_matrix(i, 2) = std::sqrt(marginals.marginalCovariance(X(i))(0,0));

      // V estimate
      result_matrix(i, 3) = result.at<Vector1>(V(i))(0);
      result_matrix(i, 4) = std::sqrt(marginals.marginalCovariance(V(i))(0,0));

      // X real
      result_matrix(i, 5) = x_real(i);
    }
  }
  std::cout << "Optimizer has finished...saving results..." << std::endl;
  save_csv(output_file, result_matrix);

  return 0;
}
