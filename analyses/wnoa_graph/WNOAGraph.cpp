#include <gtsam/nonlinear/Interpolator.h>

#include <chrono>

#include <valgrind/callgrind.h>
#include "LostInTheWoodsExample.h"
#include <gtsam/nonlinear/WNOAFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/WNOALevenbergMarquardtOptimizer.h>

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
  int n_unary;
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
    for(unsigned int m = 0; m < p.n_unary; m++)  // Add multiple priors to increase density for testing purposes
    {
      graph.addPrior(P(i), pose_meas, p.cov_diag_unary.asDiagonal());
    }

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
  cout << "Graph Conversion Time Wrapper: " << T_interp_graph << " (micro-s)" << endl;

  start = chrono::high_resolution_clock::now();
  WNOAFactorGraph<Pose2> graph_wnoa = interpolateWNOAFactorGraph<Pose2>(
    graph, estimated_states, interpolated_states, p.Q_wnoa, p.fixed_noise);
  end = chrono::high_resolution_clock::now();
  auto T_interp_graph_wnoa =
      chrono::duration_cast<chrono::microseconds>(end - start).count();
  cout << "Graph Conversion Time Cached: " << T_interp_graph_wnoa << " (micro-s)" << endl;

  // Timing test for linearization of both graphs
  start = chrono::high_resolution_clock::now();
  for(unsigned int i = 0; i < p.n_runs; i++)
  {
    graph.linearize(values_init);
  }
  end = chrono::high_resolution_clock::now();
  auto T_linearize_original =
      chrono::duration_cast<chrono::microseconds>(end - start).count()/p.n_runs;
  cout << "Original FG Linearization Time: " << T_linearize_original << " (micro-s)" << endl;


  start = chrono::high_resolution_clock::now();
  for(unsigned int i = 0; i < p.n_runs; i++)
  {
    graph_interp.linearize(values_interp_init);
  }
  end = chrono::high_resolution_clock::now();
  auto T_linearize_interp =
      chrono::duration_cast<chrono::microseconds>(end - start).count()/p.n_runs;
  cout << "Wrapper FG Linearization Time: " << T_linearize_interp << " (micro-s)" << endl;

  start = chrono::high_resolution_clock::now();
  for(unsigned int i = 0; i < p.n_runs; i++)
  {
    graph_wnoa.linearize(values_interp_init);
  }
  end = chrono::high_resolution_clock::now();
  auto T_linearize_wnoa =
      chrono::duration_cast<chrono::microseconds>(end - start).count()/p.n_runs;
  cout << "Cached FG Linearization Time: " << T_linearize_wnoa << " (micro-s)" << endl;



  auto linear_graph_interp = graph_interp.linearize(values_interp_init);
  auto linear_graph_wnoa = graph_wnoa.linearize(values_interp_init);

  // check if both graphs are identical
  std::cout << "Checking if both linearized graphs are identical..." << std::endl;
  if(linear_graph_interp->equals(*linear_graph_wnoa, 1e-9))
  {
    std::cout << "Both linearized graphs are identical!" << std::endl;
  }
  else
  {
    std::cout << "Graphs are NOT identical!" << std::endl;
  }

  // Timing test for error of both graphs


  start = chrono::high_resolution_clock::now();
  for(unsigned int i = 0; i < p.n_runs; i++)
  {
    graph.error(values_init);
  }
  end = chrono::high_resolution_clock::now();
  auto T_error_original =
      chrono::duration_cast<chrono::microseconds>(end - start).count()/p.n_runs;
  cout << "Original FG Error Time: " << T_error_original << " (micro-s)" << endl;

  start = chrono::high_resolution_clock::now();
  for(unsigned int i = 0; i < p.n_runs; i++)
  {
    graph_interp.error(values_interp_init);
  }
  end = chrono::high_resolution_clock::now();
  auto T_error_interp =
      chrono::duration_cast<chrono::microseconds>(end - start).count()/p.n_runs;
  cout << "Wrapper FG Error Time: " << T_error_interp << " (micro-s)" << endl;   

  start = chrono::high_resolution_clock::now();
  for(unsigned int i = 0; i < p.n_runs; i++)
  {
    graph_wnoa.error(values_interp_init);
  }
  end = chrono::high_resolution_clock::now();
  auto T_error_wnoa =
      chrono::duration_cast<chrono::microseconds>(end - start).count()/p.n_runs;
  cout << "Cached FG Error Time: " << T_error_wnoa << " (micro-s)" << endl;

  // Check if the returned error is the same
  double error_interp = graph_interp.error(values_interp_init);
  double error_wnoa = graph_wnoa.error(values_interp_init);
  std::cout << "Checking if both errors are identical..." << std::endl;
  if(fabs(error_interp - error_wnoa) < 1e-9)
  {
    std::cout << "Both errors are identical!" << std::endl;
  }
  else
  {
    std::cout << "Errors are NOT identical!" << std::endl;
    std::cout << "Error Interp: " << error_interp << " Error WNOA: " << error_wnoa << std::endl;
  }

  // set up optimizer
  LevenbergMarquardtParams params;
  params.setVerbosityLM("SILENT");

  // run optimization on original graph
  start = chrono::high_resolution_clock::now();
  Values result_original;
  for(unsigned int i = 0; i < p.n_runs; i++)
  {
    auto lm_opt_original = LevenbergMarquardtOptimizer(graph, values_init, params);
    result_original = lm_opt_original.optimize();
  }
  end = chrono::high_resolution_clock::now();
  auto T_solve_original =
    chrono::duration_cast<chrono::microseconds>(end - start).count()/p.n_runs;
  cout << "Original Graph Solve Time: " << T_solve_original << " micros" << endl;

  // run optimization on interpolated version
  start = chrono::high_resolution_clock::now();
  Values result_interp;
  for(unsigned int i = 0; i < p.n_runs; i++)
  {
    auto lm_opt_interp = LevenbergMarquardtOptimizer(graph_interp, values_interp_init, params);
    result_interp = lm_opt_interp.optimize();
  }
  end = chrono::high_resolution_clock::now();
  auto T_solve_interp =
    chrono::duration_cast<chrono::microseconds>(end - start).count()/p.n_runs;
  cout << "Wrapper Graph Solve Time: " << T_solve_interp << " micros" << endl;
  
  // Solve WNOA graph
  start = chrono::high_resolution_clock::now();
  Values result_wnoa;
  for(unsigned int i = 0; i < p.n_runs; i++)
  {
    auto lm_opt_wnoa = WNOALevenbergMarquardtOptimizer<Pose2>(graph_wnoa, values_interp_init, params);
    result_wnoa = lm_opt_wnoa.optimize();
  }
  end = chrono::high_resolution_clock::now();
  auto T_solve_WNOA =
    chrono::duration_cast<chrono::microseconds>(end - start).count()/p.n_runs;
  cout << "Cached Graph Solve Time: " << T_solve_WNOA << " micros" << endl;

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
  p.n_unary = config["params"]["n_unary"].as<int>();

  runInterpExample(p);
}