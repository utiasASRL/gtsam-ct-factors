#include <gtsam/nonlinear/Interpolator.h>

#include <chrono>

#include "LostInTheWoodsExample.h"

using namespace std;
using namespace gtsam;

using symbol_shorthand::P;
using symbol_shorthand::V;

Vector3 sample_vector(Vector3 cov, std::mt19937& gen) {
  std::normal_distribution<> dist(0.0, 1.0);  // mean=0, stddev=1

  // Fill Eigen vector with samples
  Vector3 vec_white;
  for (int i = 0; i < 3; ++i) {
    vec_white(i) = dist(gen);
  }
  // color the vector
  Vector3 vec = cov.cwiseSqrt().asDiagonal() * vec_white;
  return vec;
}

struct InterpExampleParams {
  Vector3 cov_diag_unary;
  Vector3 Q_wnoa;
  Vector3 vel_mean;
  int N;
  int period_interp;
  double del_t;
  bool perturb_meas;
};

void runInterpExample(InterpExampleParams& p, bool run_full_opt=true) {
  // Init graphs, values, states
  NonlinearFactorGraph graph;
  Values values_init;
  vector<StateData> states(p.N);
  Pose2 pose_init(0.0, 0.0, 0.0);

  // Init RNG
  std::random_device rd;   // Seed
  std::mt19937 gen(rd());  // Mersenne Twister RNG
  // Velocity sampled covariance
  Vector3 cov_vel = pow(p.del_t, 2) * p.Q_wnoa;

  // Define trajectory with fixed velocity
  Pose2 pose_curr;
  Vector3 velocity;
  double time = 0;
  for (int i = 0; i < p.N; i++) {
    // get current pose and velocity
    if (i == 0) {
      pose_curr = pose_init;
    } else {
      time += p.del_t;  // update time
      // update pose
      pose_curr = pose_curr.expmap(p.del_t * p.vel_mean);
      // Add WNOA motion model
      auto factor_wnoa = std::make_shared<WNOAMotionFactor<Pose2>>(
          P(i - 1), V(i - 1), P(i), V(i), p.del_t, p.Q_wnoa);
      graph.add(factor_wnoa);
    }
    // add to values
    values_init.insert(P(i), pose_curr);
    values_init.insert(V(i), velocity);
    // add prior
    Pose2 pose_meas;
    if (p.perturb_meas) {
      Vector3 pert = sample_vector(p.cov_diag_unary, gen);
      pose_meas = pose_curr.expmap(pert);
    } else {
      pose_meas = pose_curr;
    }
    graph.addPrior(P(i), pose_meas, p.cov_diag_unary.asDiagonal());

    // Track list of states
    states[i] = StateData(P(i), V(i), time);
  }

  // Set up interpolated states
  set<StateData> interpolated_states;
  set<StateData> estimated_states;
  Values values_interp_init;
  for (int i = 0; i < p.N; i++) {
    if (i == 0 || i == p.N - 1 || i % p.period_interp == 0) {
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
      graph, estimated_states, interpolated_states, p.Q_wnoa);
  
  if (run_full_opt){
    // set up optimizer
    LevenbergMarquardtParams params;
    params.setVerbosityLM("SUMMARY");
    // Solve full graph
    Values result_full =
      LevenbergMarquardtOptimizer(graph, values_init, params).optimize();
    // run optimization on interpolated version
    Values result_interp =
        LevenbergMarquardtOptimizer(graph_interp, values_interp_init, params)
            .optimize();

    // define covariance map
    auto cov_map_interp = std::make_shared<Interpolator<Pose2>::CovarianceMap>();
    // recover interpolated values and covariances
    Values result_recov =
        updateInterpValues<Pose2>(graph_interp, result_interp, estimated_states,
                                  interpolated_states, p.Q_wnoa, cov_map_interp);

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

  // compute per-iteration times for each graph
  // reduced graph
  auto opt_reduced = GaussNewtonOptimizer(graph_interp, values_interp_init);
  opt_reduced.iterate(); // run once to clear any overhead
  auto start = chrono::high_resolution_clock::now();
  opt_reduced.iterate();
  auto end = chrono::high_resolution_clock::now();
  auto duration_interp =
      chrono::duration_cast<chrono::microseconds>(end - start).count();
  cout << "Iteration Time for Interpolated Graph: " << duration_interp << " micros"
       << endl;
  // full graph
  auto opt_full = GaussNewtonOptimizer(graph, values_init);
  opt_full.iterate(); // run once to clear any overhead
  start = chrono::high_resolution_clock::now();
  opt_full.iterate();
  end = chrono::high_resolution_clock::now();
  auto duration_full =
      chrono::duration_cast<chrono::microseconds>(end - start).count();
  cout << "Iteration Time for Full Graph: " << duration_full << " micros" << endl;
}

int main(int argc, char* argv[]) {
  // Get configuration data
  bool run_full_opt = true;
  string config_file = "simple_traj/SimpleInterpExampleDefault.yaml";
  if (argc > 1) {
    run_full_opt = std::stoi(argv[1]) != 0;
  }
  if (argc > 2) {
    config_file = argv[2];
  }
  
  YAML::Node config = YAML::LoadFile(config_file);

  // Set parameters
  InterpExampleParams p;

  // diagonal of unary meas covariance
  p.cov_diag_unary =
      Vector3(config["noise"]["unary"].as<vector<double>>().data());
  // WNOA Power Spectral Density
  p.Q_wnoa = Vector3(config["noise"]["wnoa"].as<vector<double>>().data());

  p.N = config["params"]["n_points"].as<int>();  // total number of points
  p.period_interp =
      config["params"]["period_interp"]
          .as<int>();  // number of interpolated points between borders
  p.del_t = config["params"]["del_t"].as<double>();  // timestep
  p.vel_mean =
      Vector3(config["params"]["vel_mean"].as<vector<double>>().data());
  p.perturb_meas = config["flags"]["perturb_meas"].as<bool>();

  runInterpExample(p, run_full_opt);
}