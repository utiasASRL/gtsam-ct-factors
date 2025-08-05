#include "GiantGlassOfMilkExample.h"

int main() {
  // input processing 
  bool use_interpolation = true;

  string input_file = "GiantGlassOfMilk.csv";
  string output_file = "../results/milk.csv";
  
  string filename = findExampleDataFile(input_file);

  // Loading data as: t, x, v, x_real
  Matrix data = load_csv<Eigen::MatrixXd>(filename, false);

  Vector times = data.col(0);
  Vector x_meas = data.col(1);
  Vector v_meas = data.col(2);
  Vector x_real = data.col(3);

  // Defining timing variables
  double dt = 0.1;
  double dt_meas = 5.0;
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

  // Defining shorthand for symbols
  using symbol_shorthand::X; // state
  using symbol_shorthand::V; // velocity

  // Add initial guess for first state (zero)
  initialEstimate.insert(X(0),Point1(0.0));
  initialEstimate.insert(V(0),Vector1(0.0));

  // Run through all states and add WNOA prior factors between neighbouring states
  unsigned int num_states = (int)(times.tail<1>()(0) / dt_state) + 1;
  for(unsigned int i = 0; i < num_states - 1; i++)
  {
      graph.add(WNOAMotionFactor<Point1>(X(i),V(i),X(i+1),V(i+1),dt_state,Qc_mat));

      // Add initial guess for next state (zero)
      initialEstimate.insert(X(i+1),Point1(0.0));
      initialEstimate.insert(V(i+1),Vector1(0.0));
  }

  // Add measurement factors for both position and velocity
  int skip = dt_meas/dt; // Skip factor, depending on the measurement frequency

  // times_est, x_est, x_std, v_est, v_std, x_real, times_meas, x_meas, v_meas
  Matrix result_matrix = Matrix::Zero(1 + (num_states - 1) * dt_state/dt, 9);


  for(unsigned int i = 0; i < 1 + (num_states-1) * dt_state/dt; i = i + skip)
  {
    // Save used measurements for plotting later
    result_matrix(i/skip, 6) = times(i);
    result_matrix(i/skip, 7) = x_meas(i);
    result_matrix(i/skip, 8) = v_meas(i);

    Point1 position_measurement(x_meas(i));
    Vector1 velocity_measurement(v_meas(i));


    graph.add(PriorFactor<Point1>(X(i/skip),position_measurement, positionNoise));
    graph.add(PriorFactor<Vector1>(V(i/skip),velocity_measurement, velocityNoise));

  }

  // Optimize
  LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
  Values result = optimizer.optimize();
  Marginals marginals(graph, result);

  int result_mat_idx = 0;
  Interpolator<Point1> interpolator(Qc_mat);
  for(unsigned int i = 0; i < num_states; i++)
  {
    // Time
    result_matrix(result_mat_idx, 0) = i*dt_state;

    // X estimate
    result_matrix(result_mat_idx, 1) = result.at<Point1>(X(i))(0);
    result_matrix(result_mat_idx, 2) = std::sqrt(marginals.marginalCovariance(X(i))(0,0));

    // V estimate
    result_matrix(result_mat_idx, 3) = result.at<Vector1>(V(i))(0);
    result_matrix(result_mat_idx, 4) = std::sqrt(marginals.marginalCovariance(V(i))(0,0));

    // X real
    result_matrix(result_mat_idx, 5) = x_real(i * (int)(dt_state / dt));

    result_mat_idx++;

    if(i < num_states - 1) {
      // Interpolate states
      for(unsigned int j = 1; j < dt_state/dt; j++) {


        std::pair<Point1, Vector1> interpolated_state;
        std::pair<Point1, Vector1> first_state(result.at<Point1>(X(i)), result.at<Vector1>(V(i)));
        std::pair<Point1, Vector1> second_state(result.at<Point1>(X(i+1)), result.at<Vector1>(V(i+1)));

        interpolated_state = interpolator.interpolatePoseAndVelocity(first_state, i*dt_state, second_state, (i+1)*dt_state, i*dt_state + j * dt);

        result_matrix(result_mat_idx, 0) = i*dt_state + j * dt;
        result_matrix(result_mat_idx, 1) = interpolated_state.first(0);
        result_matrix(result_mat_idx, 2) = 0;
        result_matrix(result_mat_idx, 3) = interpolated_state.second(0);
        result_matrix(result_mat_idx, 4) = 0;
        result_matrix(result_mat_idx, 5) = x_real(i * (int)(dt_state / dt) + j);
        result_mat_idx++;
      }
    }

  }


  std::cout << "Optimizer has finished...saving results..." << std::endl;
  save_csv(output_file, result_matrix);

  return 0;
}
