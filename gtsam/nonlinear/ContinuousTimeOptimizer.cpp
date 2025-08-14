/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file ContinuousTimeSolver.h
 * @brief Solver for continuous-time factor graphs
 * @date August 13, 2025
 * @author Sven Lilge
 */
#include <gtsam/nonlinear/ContinuousTimeOptimizer.h>
#include <gtsam/nonlinear/internal/NonlinearOptimizerState.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/inference/Ordering.h>
#include <gtsam/base/Vector.h>
#include <gtsam/linear/GaussianBayesNet.h>
#if GTSAM_USE_BOOST_FEATURES
#include <gtsam/base/timing.h>
#include <gtsam/inference/Symbol.h>
#endif

#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>

using namespace std;



namespace gtsam {

typedef internal::NonlinearOptimizerState State;



/* ************************************************************************* */
ContinuousTimeOptimizer::ContinuousTimeOptimizer(const NonlinearFactorGraph& graph,
                                                 const Values& initialValues,
                                                 const ContinuousTimeParams& params)
    : NonlinearOptimizer(
          graph, std::unique_ptr<State>(new State(initialValues, graph.error(initialValues)))),
      params_(params) {


        interpolatedKeys_ = params_.interpolatedStates;

        // Initialize the remaining keys as the keys in the graph that are not interpolated
        for (const auto& key : graph.keys()) {
          if (std::find(params_.interpolatedStates.begin(), params_.interpolatedStates.end(), key) == params_.interpolatedStates.end()) {
            remainingKeys_.push_back(key);
          }
        }

        // Initialize the remaining graph with factors that do not involve interpolated keys
        for(auto factor : graph)
        {
          bool has_interpolated_key = false;
          for(const auto& key : factor->keys()) {
            if(std::find(params_.interpolatedStates.begin(), params_.interpolatedStates.end(), key) != params_.interpolatedStates.end()) {
              has_interpolated_key = true;
              break;
            }
          }
          if(!has_interpolated_key)
            remaining_graph_.add(factor);
        }

        //print all the keys in the remaining graph
        //cout << "Remaining graph keys: ";
        //for (const auto& key : remaining_graph_.keys()) {
        //  cout << Symbol(key).chr() << Symbol(key).index() << " ";
        //}
        //cout << endl;

        //print all the keys in the remaining graph
        //cout << "Remaining keys: ";
        //for (const auto& key : remainingKeys_) {
        //  cout << Symbol(key).chr() << Symbol(key).index() << " ";
        //}
        //cout << endl;

        //print all the keys in the remaining graph
        //cout << "Interpolated keys: ";
        //for (const auto& key : interpolatedKeys_) {
        //  cout << Symbol(key).chr() << Symbol(key).index() << " ";
        //}
        //cout << endl;
      



      }

/* ************************************************************************* */
void ContinuousTimeOptimizer::initTime() {
  // use chrono to measure time in microseconds
  startTime_ = std::chrono::high_resolution_clock::now();
}

/* ************************************************************************* */
GaussianFactorGraph::shared_ptr ContinuousTimeOptimizer::linearize() const {
  return graph_.linearize(state_->values);
}


/* ************************************************************************* */
GaussianFactorGraph::shared_ptr ContinuousTimeOptimizer::iterate() {
  auto currentState = static_cast<const State*>(state_.get());

  gttic(CT_iterate);

  // Linearize graph
  GaussianFactorGraph::shared_ptr linear = linearize();

  // Eliminate the interpolated keys from the linearized graph
  auto [bayesNet, graphAfterElimination] = linear->eliminatePartialSequential(interpolatedKeys_);

  std::vector<GaussianFactor::shared_ptr> marginalFactors;
  // Run through the factors in the graph after elimination
  // Store the new marginal factors that are not part of the original graph
  for(size_t i = 0; i < graphAfterElimination->size(); ++i) {
    auto factor = (*graphAfterElimination)[i];
    bool found = false;
    // Check if the factor is not in the original graph
    for(size_t j = 0; j < linear->size(); ++j) {
      if((*linear)[j] == factor) {
        found = true;
        break;
      }
    }
    // If we reach here, the factor is not in the original graph
    if (!found) {
      marginalFactors.push_back(factor);
    }
  }

  Values newValues = currentState->values;
  VectorValues delta, total_delta;
  for(int i = 0; i < params_.maxInnerIterations; i++)
  {
    // Linearize the remaining nonlinear graph around current values
    auto linear_remaining_graph = remaining_graph_.linearize(newValues);

    // Add in the marginal factors from the previous elimination
    for (const auto& f : marginalFactors) {
      linear_remaining_graph->add(f);
    }

    // Solve the linear system
    delta = linear_remaining_graph->optimize();

    // update the newValues given a delta
    newValues = newValues.retract(delta);
    total_delta = currentState->values.localCoordinates(newValues);

    if(total_delta.norm() > params_.deltaThreshold) {
      break;
    }
  }

  //Print keys in delta and total_delta
  //cout << "Delta keys: ";
  //for (const auto& [key, value] : delta) {
  //  cout << Symbol(key).chr() << Symbol(key).index() << " ";
  //}
  //cout << endl;
  //cout << "Total delta keys: ";
  //for (const auto& [key, value] : total_delta) {
  //  cout << Symbol(key).chr() << Symbol(key).index() << " ";
  //}
  //cout << endl;

  VectorValues delta_remaining;
  // Extract only the delta values for the remaining keys
  // Do not use the delta of the interpolated keys, we will get those via the bayesNet
  for (const auto& key : remainingKeys_) {
    if (total_delta.exists(key)) {
      delta_remaining.insert(key, total_delta.at(key));
    }
  }

  // Add in the delta of the interpolated keys using the bayesNet
  VectorValues delta_all_states = bayesNet->optimize(delta_remaining);

  //Update the state values and the error
  newValues = currentState->values.retract(delta_all_states);
  auto newError = graph_.error(newValues);

  state_ = std::make_unique<State>(newValues, newError,state_->iterations + 1);

  // Return the linearized factor graph form the beginning of the iteration
  return linear;
}

} /* namespace gtsam */

