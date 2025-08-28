#include <gtsam/nonlinear/Interpolator.h>

#include "LostInTheWoodsExample.h"

using namespace std;
using namespace gtsam;

using symbol_shorthand::P;
using symbol_shorthand::V;

void runInterpExample(Vector& cov_diag_unary, Vector& Q_wnoa, int n_points,
                      int period_interp, double del_t, Vector3& velocity,
                      bool include_pose_prior, bool include_vel_prior) {
  // Init graphs, values, states
  NonlinearFactorGraph graph;
  Values values_init;
  vector<StateData> states(n_points);
  Pose2 pose_init(0.0, 0.0, 0.0);

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
    if (include_pose_prior) {
      graph.addPrior(P(i), pose_curr, cov_diag_unary.asDiagonal());
    }
    if (include_vel_prior) {
      graph.addPrior(V(i), velocity, cov_diag_unary.asDiagonal());
    }
    // Track list of states
    states[i] = StateData(P(i), V(i), time);
  }
  // prior on first and last state when using velocity
  if (include_vel_prior && !include_pose_prior) {
    graph.addPrior(P(0), pose_init, cov_diag_unary.asDiagonal());
    graph.addPrior(P(n_points - 1), pose_init, cov_diag_unary.asDiagonal());
  }
  // set up optimizer
  LevenbergMarquardtParams params;
  params.setVerbosityLM("SUMMARY");
  // Solve full graph
  Values result_full =
      LevenbergMarquardtOptimizer(graph, values_init, params).optimize();

  // Set up interpolated states
  set<StateData> interpolated_states;
  set<StateData> estimated_states;
  Values values_interp_init;
  for (int i = 0; i < n_points; i++) {
    if (i == 0 || i == n_points - 1 || i % period_interp == 0) {
      estimated_states.insert(states[i]);
      // fill in initial values
      Key pose_key = states[i].pose;
      Key vel_key = states[i].vel;
      values_interp_init.insert(pose_key, values_init.at<Pose2>(pose_key));
      values_interp_init.insert(vel_key, values_init.at<Vector3>(vel_key));
    } else {
      interpolated_states.insert(states[i]);
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
  Values result_recov =
      updateInterpValues<Pose2>(graph_interp, result_interp, estimated_states,
                                interpolated_states, Q_wnoa, cov_map_interp);

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
  string config_file = "examples/SimpleInterpExampleDefault.yaml";
  if (argc > 1) {
    config_file = argv[1];
  }
  YAML::Node config = YAML::LoadFile(config_file);

  // diagonal of unary meas covariance
  Vector cov_diag_unary =
      Vector3(config["noise"]["unary"].as<vector<double>>().data());
  // WNOA Power Spectral Density
  Vector Q_wnoa = Vector3(config["noise"]["wnoa"].as<vector<double>>().data());

  int n_points =
      config["params"]["n_points"].as<int>();  // total number of points
  int period_interp =
      config["params"]["period_interp"]
          .as<int>();  // number of interpolated points between borders
  double del_t = config["params"]["del_t"].as<double>();  // timestep
  Vector3 velocity =
      Vector3(config["params"]["velocity"].as<vector<double>>().data());
  Pose2 pose_init(0.0, 0.0, 0.0);
  bool include_pose_prior = config["flags"]["pose_prior"].as<bool>();
  bool include_vel_prior = config["flags"]["vel_prior"].as<bool>();

  runInterpExample(cov_diag_unary, Q_wnoa, n_points, period_interp, del_t,
                   velocity, include_pose_prior, include_vel_prior);
}