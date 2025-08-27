#include <gtsam/nonlinear/Interpolator.h>

#include "LostInTheWoodsExample.h"

using namespace std;
using namespace gtsam;

using symbol_shorthand::P;
using symbol_shorthand::V;

void runInterpExample() {
  // Define Noise Models
  Vector cov_diag_unary = Vector3::Ones();  // diagonal of unary meas covariance
  Vector Q_wnoa = Vector3::Ones();  // WNOA Power Spectral Density

  // Parameters for trajectory
  int n_points = 100;       // total number of points
  int period_interp = 25;  // number of interpolated points between borders
  double del_t = 0.1;      // timestep
  Vector3 velocity;
  velocity << 1.0, 0.0, 0.1;
  Pose2 pose_init(0.0, 0.0, 0.0);
  bool include_pose_prior = true;
  bool include_vel_prior = false;

  // Init graphs, values, states
  NonlinearFactorGraph graph;
  Values values_init;
  vector<StateData> states(n_points);

  // Define trajectory with fixed velocity
  Pose2 pose_curr;
  double time = 0;
  for (int i = 0; i < n_points; i++) {
    // get current pose
    if (i == 0) {
      pose_curr = pose_init;
    } else {
      time += del_t;                                   // update time
      pose_curr = pose_curr.expmap(del_t * velocity);  // update pose
      // Add WNOA motion model
      auto factor_wnoa = std::make_shared<WNOAMotionFactor<Pose2>>(
          P(i - 1), V(i - 1), P(i), V(i), del_t, Q_wnoa);
      graph.add(factor_wnoa);
    }
    // add to values
    values_init.insert(P(i), pose_curr);
    values_init.insert(V(i), velocity);
    // add prior
    if (include_pose_prior){
      graph.addPrior(P(i), pose_curr, cov_diag_unary.asDiagonal());
    }
    if (include_vel_prior){
      graph.addPrior(V(i), velocity, cov_diag_unary.asDiagonal());
    }
    // Track list of states
    states[i] = StateData(P(i), V(i), time);
  }
  // prior on first state
  if (include_vel_prior && !include_pose_prior){
    graph.addPrior(P(0), pose_init, cov_diag_unary.asDiagonal());
    graph.addPrior(P(n_points-1), pose_init, cov_diag_unary.asDiagonal());
  }
  // set up optimizer
  LevenbergMarquardtParams params;
  params.setVerbosityLM("SUMMARY");
  // Solve full graph
  Values result_full =
      LevenbergMarquardtOptimizer(graph, values_init, params).optimize();
  // Save results

  // Set up interpolated states
  vector<StateData> interpolated_states;
  vector<StateData> estimated_states;
  Values values_interp_init;
  for (int i = 0; i < n_points; i++) {
    if (i == 0 || i == n_points - 1 || i % period_interp == 0) {
      estimated_states.push_back(states[i]);
      Key pose_key = states[i].pose;
      Key vel_key = states[i].vel;
      values_interp_init.insert(pose_key, values_init.at<Pose2>(pose_key));
      values_interp_init.insert(vel_key, values_init.at<Vector3>(vel_key));
    } else {
      interpolated_states.push_back(states[i]);
    }
  }
  // generate interpolated graph
  NonlinearFactorGraph graph_interp = interpolateFactorGraph<Pose2>(
      graph, estimated_states, interpolated_states, Q_wnoa);
  // run optimization on interpolated version
  Values result_interp =
      LevenbergMarquardtOptimizer(graph_interp, values_interp_init, params)
          .optimize();

  // define covariance map
  auto cov_map_interp = std::make_shared<Interpolator<Pose2>::CovarianceMap>();
  // recover interpolated values and covariances
  Values result_recov = updateInterpValues<Pose2>(
      graph_interp, result_interp, estimated_states, interpolated_states,
      Q_wnoa, cov_map_interp);

  // Save the results to files
  // Full solve
  saveResultToFile(result_full, graph, "results/simple_ex_full.csv");
  // Just estimated states
  saveResultToFile(result_interp, graph_interp, "results/simple_ex_estim.csv");
  // All states, with covariance from graph
  saveResultToFile(result_recov, graph, "results/simple_ex_interp_graph.csv");
  // All states, with covariance recovered from interpolation.
  saveResultToFile(result_recov, graph_interp, "results/simple_ex_interp.csv",
                   false, cov_map_interp);
}

int main(int argc, char* argv[]) {
  // Get configuration data
  //   string config_file = "examples/SimpleInterpEx.yaml";
  //   if (argc > 1) {
  //     config_file = argv[1];
  //   }
  //   YAML::Node config = YAML::LoadFile(config_file);
  runInterpExample();
}