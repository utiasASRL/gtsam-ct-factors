#include "LostInTheWoodsISAMExample.h"

int main(int argc, char* argv[]) {
  // Get configuration data
  string config_file = "examples/Data/LostInTheWoods.yaml";
  if (argc > 1) {
    config_file = argv[1];
  }
  YAML::Node config = YAML::LoadFile(config_file);

  // Load Files
  string input_file = config["files"]["input"].as<string>();
  string output_file = config["files"]["output"].as<string>();
  string gt_output_file = config["files"]["gt_out"].as<string>();
  // Load dataset
  DatasetLoader data;
  data.loadFromFile(input_file);
  data.checkSizes();

  // switches for factors/init
  bool include_prior = config["flags"]["prior"].as<bool>();
  bool include_odom = config["flags"]["odom"].as<bool>();
  bool include_wnoa = config["flags"]["wnoa"].as<bool>();
  bool include_br_meas = config["flags"]["br"].as<bool>();
  bool gt_init = config["flags"]["gt_init"].as<bool>();
  // Get inputs from param file
  double r_max = config["params"]["r_max"].as<double>();
  double del_t = config["params"]["del_t"].as<double>();
  int start = config["params"]["start"].as<int>();
  int end = config["params"]["end"].as<int>();
  // Get noise model parameters
  Vector sigma_prior =
      Vector3(config["noise"]["prior"].as<vector<double>>().data());
  Vector sigma_wnoa =
      Vector3(config["noise"]["wnoa"].as<vector<double>>().data());
  double sigma_y_odom = config["noise"]["odom_y"].as<double>();
  double mult_bearing = config["noise"]["bearing"].as<double>();
  double mult_range = config["noise"]["range"].as<double>();
  Vector sigma_odom =
      Vector3(sqrt(data.v_var), sigma_y_odom, sqrt(data.om_var)) * del_t;
  Vector sigma_br =
      Vector2(sqrt(mult_bearing * data.b_var), sqrt(mult_range * data.r_var));

  // Generate noise models
  auto priorNoise = noiseModel::Diagonal::Sigmas(sigma_prior);  // prior
  auto odoNoise = noiseModel::Diagonal::Sigmas(sigma_odom);     // odometry
  auto measNoise =
      noiseModel::Diagonal::Sigmas(sigma_br);  // range-bearing noise
  // Create a factor graph
  ExpressionFactorGraph graph;

  // Starting point
  Pose2 startPose(data.x_true[start], data.y_true[start], data.th_true[start]);
  // Initial Pose Prior
  if (include_prior) {
    cout << "Adding Prior on start pose: " << sigma_prior << endl;
    graph.add(PriorFactor<Pose2>(Symbol('x', start), startPose, priorNoise));
    if (include_wnoa) {
      // Add in velocity prior on first state
      cout << "Adding Prior on start velocity" << endl;
      Vector vel_init = Vector3(data.v[start], 0.0, data.om[start]);
      graph.addPrior<Vector3>(Symbol('v', start), vel_init, odoNoise);
    }
  }
  // Odometry factors
  if (include_odom) {
    cout << "Adding odometry prior factors " << endl;

    for (int i = start + 1; i <= end; i++) {
      // define odometry measurement
      Pose2 odom(data.v[i - 1] * del_t, 0.0, data.om[i - 1] * del_t);
      // add factor to graph
      const auto factor = BetweenFactor<Pose2>(Symbol('x', i - 1), Symbol('x', i), odom, odoNoise);
      graph.add(factor);
    }
  }

  // White-Noise-On-Acceleration Prior
  if (include_wnoa) {
    cout << "Adding WNOA factors" << endl;
    // Add WNOA Motion Factors between states
    for (int i = start + 1; i <= end; i++) {
      graph.add(WNOAMotionFactor<Pose2>(Symbol('x', i - 1), Symbol('v', i - 1),
                                        Symbol('x', i), Symbol('v', i), del_t,
                                        sigma_wnoa));
    }
  }

  // BearingRange Measurements
  if (include_br_meas) {
    cout << "Adding bearing range measurement factors" << endl;

    // Define landmarks
    vector<Point2> landmarks(data.n_landmarks);
    for (int j = 0; j < data.n_landmarks; j++) {
      landmarks[j] = data.landmarks.row(j);
    }

    Pose2 T_vs(data.d, 0.0, 0.0);
    for (int i = start; i <= end; i++) {
      // Define Key
      Key xi = Symbol('x', i);
      for (int j = 0; j < data.n_landmarks; j++) {
        // Check if we have a valid measurement
        if ((data.range(i, j) > 0.0) && (abs(data.bearing(i, j)) > 0.0) &&
            (data.range(i, j) < r_max)) {
          // Get Bearing Range measurement
          BearingRange2 measurement(Rot2(data.bearing(i, j)), data.range(i, j));
          // Compute the bearing and range Prediction
          auto predict = BearingRangeLandmarkPrediction(xi, landmarks[j], T_vs);
          // Define Factor
          graph.addExpressionFactor(predict, measurement, measNoise);
        }
      }
    }
  }
  // Get ground truth solution
  Values gt;
  for (int i = start; i <= end; i++) {
    gt.insert(Symbol('x', i),
              Pose2(data.x_true[i], data.y_true[i], data.th_true[i]));
    if (include_wnoa) {
      gt.insert(Symbol('v', i), Vector3(data.v[i], 0.0, data.om[i]));
    }
  }
  // Initialization
  Values currentEstimate;
  if (gt_init) {
    cout << "Ground truth initialization enabled" << endl;
    currentEstimate = gt;
  } else {
    cout << "Ground truth initialization disabled" << endl;
    // Rollout odometry
    for (int i = start; i <= end; i++) {
      if (i == start) {
        currentEstimate.insert(Symbol('x', i), startPose);
        Vector3 zero = Vector3::Zero();
        if (include_wnoa) {
          currentEstimate.insert(Symbol('v', i), zero);
        }
      } else {
        Vector vel = Vector3(data.v[i - 1], 0.0, data.om[i - 1]);
        Vector3 vel_t = del_t * vel;
        Pose2 odom = Pose2::Expmap(vel_t);
        currentEstimate.insert(Symbol('x', i),
                       currentEstimate.at<Pose2>(Symbol('x', i - 1)).compose(odom));
        if (include_wnoa) {
          currentEstimate.insert(Symbol('v', i), vel);
        }
      }
    }
  }

  // Collect some states that will be considered "interpolated"
  KeyVector interpolatedKeys; //interpolated keys, which we will eliminate
  KeyVector remainingKeys; // these keys will remain in the graph
  int M = 3;
  for (int i = start; i <= end; i++) {

    if((i-start)%M != 0 && i != end) {
      interpolatedKeys.push_back(Symbol('x', i));
      interpolatedKeys.push_back(Symbol('v', i));
    }
    else {
      remainingKeys.push_back(Symbol('x', i));
      remainingKeys.push_back(Symbol('v', i));
    }
  }


  auto start_time = std::chrono::high_resolution_clock::now();

  // construct graphs that contain factors with involve interpolated keys
  // and those that do not
  // This is necessary because we will eliminate the interpolated keys
  // and we want to keep the remaining factors in the graph.
  ExpressionFactorGraph graph_interpolated, graph_remaining;
  for(auto factor : graph)
  {
    bool has_interpolated_key = false;
    for(const auto& key : factor->keys()) {
      if(std::find(interpolatedKeys.begin(), interpolatedKeys.end(), key) != interpolatedKeys.end()) {
        has_interpolated_key = true;
        break;
      }
    }
    if(has_interpolated_key)
      graph_interpolated.add(factor);
    else
      graph_remaining.add(factor);
  }

  std::cout << "Original graph has " << graph.size() << " factors." << std::endl;
  std::cout << "Graph with interpolated keys has " << graph_interpolated.size() << " factors." << std::endl;
  std::cout << "Graph with remaining keys has " << graph_remaining.size() << " factors." << std::endl;

 
  
  graph.saveGraph("results/lost_graph_full.dot", currentEstimate);
  graph_interpolated.saveGraph("results/lost_graph_inter.dot", currentEstimate);
  graph_remaining.saveGraph("results/lost_graph_remain.dot", currentEstimate);

  for(unsigned int iter = 0; iter < 100; iter++)
  {
    std::cout << "Iteration " << iter << "..." << std::endl;
    // We now linearize the full graph at the current estimate
    auto linearGraph = graph.linearize(currentEstimate);

    // we now eliminate the interpolated keys from the linearized graph
    // this gives us a bayes net and a remaining graph
    auto [bayesNet, remainingGraph] = linearGraph->eliminatePartialSequential(interpolatedKeys);

    if(iter == 0)
    {
      bayesNet->saveGraph("results/lost_bayes_net.dot");
      remainingGraph->saveGraph("results/lost_remaining_graph.dot");
    }

    // Store a set of original factors
    std::vector<GaussianFactor::shared_ptr> originalFactors;
    for (size_t i = 0; i < linearGraph->size(); ++i) {
      originalFactors.push_back((*linearGraph)[i]);
    }

    // Remaining graph after elimination
    std::vector<GaussianFactor::shared_ptr> newFactors;
    for (size_t i = 0; i < remainingGraph->size(); ++i) {
      newFactors.push_back((*remainingGraph)[i]);
    }

    // Detect which of the new factors are different from the original factors and only keep those
    std::vector<GaussianFactor::shared_ptr> differentFactors;
    for (const auto& newFactor : newFactors) {
      bool found = false;
      for (const auto& originalFactor : originalFactors) {
        if (newFactor == originalFactor) {
          found = true;
          break;
        }
      }
      if (!found) {
        differentFactors.push_back(newFactor);
      }
    }

    Values updatedValues = currentEstimate;
    // Run a couple iterations on the smaller graph, adding in the fixed factors from elimination of the interpolated keys
    VectorValues total_delta;
    for(int i = 0; i < 10; i++)
    {
      std::cout << "Inner iteration " << i << "..." << std::endl;
      auto linear_remaining_graph = graph_remaining.linearize(updatedValues);

      // add in the different factors from the elimination
      for(const auto& factor : differentFactors)
      {
        linear_remaining_graph->add(*factor);
      }


      if(iter == 0 && i == 0) {
        linear_remaining_graph->saveGraph("results/lost_graph_small.dot");
      }

      // solve linear system
      auto delta = linear_remaining_graph->optimize();

      // update current estimate given a delta
      updatedValues = updatedValues.retract(delta);
      total_delta = currentEstimate.localCoordinates(updatedValues);

      //if (total_delta.norm() > 0.05) {
      //  break;
      //}

    }

    Values newEstimate;
    // Run through all vales in updatedValues except last one
    for (size_t i = 0; i < updatedValues.size() - 1; i++) {
      // print all keys and values
      Key currentKey = updatedValues.keys()[i];
      std::cout << "Key: " << currentKey << std::endl;
      // If you want to print the value, you need to handle the type explicitly.
      // For example, if you expect Pose2 or Vector3, you can do:
      if (updatedValues.exists(currentKey)) {
        try {
          const Pose2& pose = updatedValues.at<Pose2>(currentKey);
          std::cout << "Pose2 Value: " << pose << std::endl;
        } catch (...) {
          try {
        const Vector3& vec = updatedValues.at<Vector3>(currentKey);
        std::cout << "Vector3 Value: " << vec.transpose() << std::endl;
          } catch (...) {
        std::cout << "Unknown value type for key: " << currentKey << std::endl;
          }
        }
      }
    }

    //Get vector values from current estimate
    VectorValues delta = total_delta;

    // Extract only the values for the remaining keys
    VectorValues deltaRemaining;
    for (const auto& key : remainingKeys) {
      if (delta.exists(key)) {
        deltaRemaining.insert(key, delta.at(key));
      }
    }

    //Use this to get the delta for the interpolated keys
    VectorValues deltaInterpolated = bayesNet->optimize(deltaRemaining);


    std::cout << "Norm of delta: " << deltaInterpolated.norm() << std::endl;

    //Update the interpolated keys in the current estimate
    currentEstimate = currentEstimate.retract(deltaInterpolated);

    if(deltaInterpolated.norm() < 1e-6) {
      std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
      break;
    }

  }

  Values result = currentEstimate;


  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  cout << "Optimization took " << duration << " milliseconds." << endl;

  // Save results
  cout << "Optimizer has finished...saving results..." << endl;
  saveResultToFile(result, graph, output_file);
  saveResultToFile(gt, graph, "results/lost_gt.csv");

  return 0;
}
