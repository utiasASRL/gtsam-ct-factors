#include "LostInTheWoodsExample.h"

using namespace std;
using namespace gtsam;

using symbol_shorthand::P;
using symbol_shorthand::V;

void runInterpExample() {
  // Define Noise Models
  Vector sigma_unary = Vector3::Ones();  // diagonal of unary meas
  Vector sigma_wnoa = Vector3::Ones();       // WNOA Power Spectral Density

  // Parameters for trajectory
  int n_points = 100;      // total number of points
  int period_interp = 20;  // number of interpolated points between borders
  double del_t = 0.1;      // timestep
  Vector3 velocity;
  velocity << 1.0, 0.0, 0.0;
  Pose2 pose_init(0.0, 0.0, 0.0);

  // Init graphs, values, states
  NonlinearFactorGraph graph;
  Values values_gt;
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
          P(i - 1), V(i - 1), P(i), V(i), del_t, sigma_wnoa);
      graph.add(factor_wnoa);
    }
    // add to values
    values_gt.insert(P(i), pose_curr);
    values_gt.insert(V(i), velocity);
    // add prior
    graph.addPrior(P(i), pose_curr, sigma_unary.asDiagonal());
    // Track list of states
    states[i] = StateData(P(i), V(i), time);
  }
  // set up optimizer
  LevenbergMarquardtParams params;
  params.setVerbosityLM("SUMMARY");
  // Solve full graph
  Values result_full =
      LevenbergMarquardtOptimizer(graph, values_gt, params).optimize();
  // Save results

  // Set up interpolated states
  vector<StateData> interpolated_states;
  vector<StateData> estimated_states;
  for (int i = 0; i < n_points; i++) {
    if (i == 0 || i == n_points - 1 || i % period_interp == 0) {
      estimated_states.push_back(states[i]);
    } else {
      interpolated_states.push_back(states[i]);
    }
  }
  // generate interpolated graph
  NonlinearFactorGraph graph_interp = interpolateFactorGraph<Pose2>(
      graph, estimated_states, interpolated_states, sigma_wnoa);
  // run optimization on interpolated version
  Values result_interp =
      LevenbergMarquardtOptimizer(graph_interp, values_gt, params).optimize();
  // recover interpolated values
  Values result_recov =
      updateInterpValues<Pose2>(graph_interp, result_interp, estimated_states,
                                interpolated_states, sigma_wnoa);

  // Save the results to files
  saveResultToFile(result_full, graph, "results/simple_ex_full.csv");
  saveResultToFile(result_interp, graph_interp, "results/simple_ex_interp.csv");
  saveResultToFile(result_recov, graph, "results/simple_ex_recov.csv");
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