#include <gtsam/nonlinear/Interpolator.h>

#include <chrono>

#include <valgrind/callgrind.h>
#include "LostInTheWoodsExample.h"
#include <gtsam/nonlinear/WNOAFactorGraph.h>

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
  bool fixed_noise;
  int n_runs;
};

void runInterpExample(InterpExampleParams& p) {
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

  // initialize clocks
  auto start = chrono::high_resolution_clock::now();
  auto end = chrono::high_resolution_clock::now();


  // generate interpolated graph
  start = chrono::high_resolution_clock::now();
  NonlinearFactorGraph graph_interp;
  graph_interp = interpolateFactorGraph<Pose2>(
    graph, estimated_states, interpolated_states, p.Q_wnoa, p.fixed_noise);
  end = chrono::high_resolution_clock::now();
  auto T_interp_graph =
      chrono::duration_cast<chrono::microseconds>(end - start).count();
  cout << "Graph Conversion Time Vanilla: " << T_interp_graph << " (micro-s)" << endl;

  start = chrono::high_resolution_clock::now();
  WNOAFactorGraph<Pose2> graph_wnoa = interpolateWNOAFactorGraph<Pose2>(
    graph, estimated_states, interpolated_states, p.Q_wnoa, p.fixed_noise);
  end = chrono::high_resolution_clock::now();
  auto T_interp_graph_wnoa =
      chrono::duration_cast<chrono::microseconds>(end - start).count();
  cout << "Graph Conversion Time WNOA FG: " << T_interp_graph_wnoa << " (micro-s)" << endl;

  // time linearization of both graphs
  start = chrono::high_resolution_clock::now();
  for(unsigned int i = 0; i < p.n_runs; i++)
  {
    graph_interp.linearize(values_interp_init);
  }
  end = chrono::high_resolution_clock::now();
  auto T_linearize_interp =
      chrono::duration_cast<chrono::microseconds>(end - start).count()/p.n_runs;
  cout << "Interp FG Linearization Time: " << T_linearize_interp << " (micro-s)" << endl;

  start = chrono::high_resolution_clock::now();
  for(unsigned int i = 0; i < p.n_runs; i++)
  {
    graph_wnoa.linearize(values_interp_init);
  }
  end = chrono::high_resolution_clock::now();
  auto T_linearize_wnoa =
      chrono::duration_cast<chrono::microseconds>(end - start).count()/p.n_runs;
  cout << "WNOA FG Linearization Time: " << T_linearize_wnoa << " (micro-s)" << endl;

  // set up optimizer
  GaussNewtonParams params;
  params.verbosity = NonlinearOptimizerParams::Verbosity::SILENT;

  // run optimization on interpolated version
  start = chrono::high_resolution_clock::now();
  Values result_interp;
  for(unsigned int i = 0; i < p.n_runs; i++)
  {
    result_interp = GaussNewtonOptimizer(graph_interp, values_interp_init, params).optimize();
  }
  end = chrono::high_resolution_clock::now();
  auto T_solve_interp =
    chrono::duration_cast<chrono::microseconds>(end - start).count()/p.n_runs;
  cout << "Interp Solve Time: " << T_solve_interp << " micros" << endl;
  // print number of iterations and resulting error
  auto gn_opt_interp = GaussNewtonOptimizer(graph_interp, values_interp_init, params);
  Values result_check = gn_opt_interp.optimize();
  cout << "Number of iterations: " << gn_opt_interp.iterations() << endl;
  cout << "Final error: " << gn_opt_interp.error() << endl;
  
  // Solve WNOA graph
  start = chrono::high_resolution_clock::now();
  Values result_full;
  for(unsigned int i = 0; i < p.n_runs; i++)
  {
    result_full = GaussNewtonOptimizer(graph_wnoa, values_interp_init, params).optimize();
  }
  end = chrono::high_resolution_clock::now();
  auto T_solve_WNOA =
    chrono::duration_cast<chrono::microseconds>(end - start).count()/p.n_runs;
  cout << "WNOA Graph Solve Time: " << T_solve_WNOA << " micros" << endl;
  // print number of iterations and resulting error
  auto gn_opt_wnoa = GaussNewtonOptimizer(graph_wnoa, values_init, params);
  Values result_check_wnoa = gn_opt_wnoa.optimize();
  cout << "Number of iterations: " << gn_opt_wnoa.iterations() << endl;
  cout << "Final error: " << gn_opt_wnoa.error() << endl;

  // define covariance map
  auto cov_map_interp =
      std::make_shared<Interpolator<Pose2>::CovarianceMap>();
  // recover interpolated values and covariances
  Values result_recov = updateInterpValues<Pose2>(
      graph_interp, result_interp, estimated_states, interpolated_states,
      p.Q_wnoa, cov_map_interp);

}

int main(int argc, char* argv[]) {
  // Get configuration data
  string config_file = "wnoa_graph/WNOAGraph.yaml";
  if (argc > 1) {
    config_file = argv[1];
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
  p.fixed_noise = config["flags"]["fixed_noise"].as<bool>();
  p.n_runs = config["params"]["n_runs"].as<int>();

  runInterpExample(p);
}