/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    WNOALevenbergMarquardtOptimizer.cpp
 * @brief   Template instantiations for WNOALevenbergMarquardtOptimizer
 * @author  Sven Lilge
 * @date    Sep 9, 2025
 */

#include <gtsam/nonlinear/WNOALevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/internal/LevenbergMarquardtState.h>
#include <gtsam/linear/linearExceptions.h>
#include <iostream>
#include <iomanip>
#include <limits>
#include <chrono>
#include <fstream>

#if GTSAM_USE_BOOST_FEATURES
#include <gtsam/base/timing.h>
#endif

namespace gtsam {

typedef internal::LevenbergMarquardtState State;
using namespace std;

// Constructor implementations  
template<typename PoseType>
WNOALevenbergMarquardtOptimizer<PoseType>::WNOALevenbergMarquardtOptimizer(
    const WNOAFactorGraph<PoseType>& graph, 
    const Values& initialValues,
    const LevenbergMarquardtParams& params)
  : LevenbergMarquardtOptimizer(graph, initialValues, params), // Pass the graph normally for now
    wnoa_graph_(const_cast<WNOAFactorGraph<PoseType>&>(graph)),
    params_(LevenbergMarquardtParams::EnsureHasOrdering(params, graph)) {
}

template<typename PoseType>
WNOALevenbergMarquardtOptimizer<PoseType>::WNOALevenbergMarquardtOptimizer(
    const WNOAFactorGraph<PoseType>& graph, 
    const Values& initialValues,
    const Ordering& ordering,
    const LevenbergMarquardtParams& params)
  : LevenbergMarquardtOptimizer(graph, initialValues, ordering, params), // Pass the graph normally for now  
    wnoa_graph_(const_cast<WNOAFactorGraph<PoseType>&>(graph)),
    params_(LevenbergMarquardtParams::ReplaceOrdering(params, ordering)) {
}

// Implementation of tryLambda that uses WNOA graph for error calculation
template<typename PoseType>
bool WNOALevenbergMarquardtOptimizer<PoseType>::tryLambda(const GaussianFactorGraph& linear,
                                            const VectorValues& sqrtHessianDiagonal) {
  auto currentState = static_cast<const internal::LevenbergMarquardtState*>(this->state_.get());
  bool verbose = (params_.verbosityLM >= LevenbergMarquardtParams::TRYLAMBDA);

#if GTSAM_USE_BOOST_FEATURES
#ifdef GTSAM_USING_NEW_BOOST_TIMERS
  boost::timer::cpu_timer lamda_iteration_timer;
  lamda_iteration_timer.start();
#else
  boost::timer lamda_iteration_timer;
  lamda_iteration_timer.restart();
#endif
#else
  auto start = std::chrono::high_resolution_clock::now();
#endif

  if (verbose)
    std::cout << "trying lambda = " << currentState->lambda << std::endl;

  // Build damped system for this lambda (adds prior factors that make it like gradient descent)
  auto dampedSystem = this->buildDampedSystem(linear, sqrtHessianDiagonal);

  // Try solving
  double modelFidelity = 0.0;
  bool step_is_successful = false;
  bool stopSearchingLambda = false;
  double newError = std::numeric_limits<double>::infinity();
  double costChange = 0.0;
  Values newValues;
  VectorValues delta;

  bool systemSolvedSuccessfully;
  try {
    // ============ Solve is where most computation happens !! =================
    delta = solve(dampedSystem, params_);
    systemSolvedSuccessfully = true;
  } catch (const IndeterminantLinearSystemException&) {
    systemSolvedSuccessfully = false;
  }

  if (systemSolvedSuccessfully) {
    if (verbose)
      std::cout << "linear delta norm = " << delta.norm() << std::endl;
    if (params_.verbosityLM >= LevenbergMarquardtParams::TRYDELTA)
      delta.print("delta");

    // Compute the old linearized error as it is not the same
    // as the nonlinear error when robust noise models are used.
    double oldLinearizedError = linear.error(VectorValues::Zero(delta));
    double newlinearizedError = linear.error(delta);

    // cost change in the linearized system (old - new)
    double linearizedCostChange = oldLinearizedError - newlinearizedError;
    if (verbose)
      std::cout << "newlinearizedError = " << newlinearizedError
           << "  linearizedCostChange = " << linearizedCostChange << std::endl;

    if (linearizedCostChange >= 0) {  // step is valid
      // update values
      gttic(retract);
      // ============ This is where the solution is updated ====================
      newValues = currentState->values.retract(delta);
      // =======================================================================
      gttoc(retract);

      // compute new error - THIS IS THE KEY LINE THAT USES OUR CUSTOM GRAPH
      gttic(compute_error);
      if (verbose)
        std::cout << "calculating error:" << std::endl;
      newError = wnoa_graph_.error(newValues); // Use WNOA graph instead of graph_
      gttoc(compute_error);

      if (verbose)
        std::cout << "old error (" << currentState->error << ") new (tentative) error (" << newError
             << ")" << std::endl;

      // cost change in the original, nonlinear system (old - new)
      costChange = currentState->error - newError;

      if (linearizedCostChange > std::numeric_limits<double>::epsilon() * oldLinearizedError) {
        // the (linear) error has to decrease to satisfy this condition
        // fidelity of linearized model VS original system between
        modelFidelity = costChange / linearizedCostChange;
        // if we decrease the error in the nonlinear system and modelFidelity is above threshold
        step_is_successful = modelFidelity > params_.minModelFidelity;
        if (verbose)
          std::cout << "modelFidelity: " << modelFidelity << std::endl;
      }  // else we consider the step non successful and we either increase lambda or stop if error
         // change is small

      double minAbsoluteTolerance = params_.relativeErrorTol * currentState->error;
      // if the change is small we terminate
      if (std::abs(costChange) < minAbsoluteTolerance) {
        if (verbose)
          std::cout << "abs(costChange)=" << std::abs(costChange)
               << "  minAbsoluteTolerance=" << minAbsoluteTolerance
               << " (relativeErrorTol=" << params_.relativeErrorTol << ")" << std::endl;
        stopSearchingLambda = true;
      }
    }
  } // if (systemSolvedSuccessfully)

  if (params_.verbosityLM == LevenbergMarquardtParams::SUMMARY) {
#if GTSAM_USE_BOOST_FEATURES
// do timing
#ifdef GTSAM_USING_NEW_BOOST_TIMERS
    double iterationTime = 1e-9 * lamda_iteration_timer.elapsed().wall;
#else
    double iterationTime = lamda_iteration_timer.elapsed();
#endif
#else
    auto end = std::chrono::high_resolution_clock::now();
    double iterationTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6;
#endif
    if (currentState->iterations == 0) {
      std::cout << "iter      cost      cost_change    lambda  success iter_time" << std::endl;
    }
    std::cout << std::setw(4) << currentState->iterations << " " << std::setw(12) << newError << " " << std::setw(12) << std::setprecision(2)
         << costChange << " " << std::setw(10) << std::setprecision(2) << currentState->lambda << " " << std::setw(6)
         << systemSolvedSuccessfully << " " << std::setw(10) << std::setprecision(2) << iterationTime << std::endl;
  }
  if (step_is_successful) {
    // we have successfully decreased the cost and we have good modelFidelity
    // NOTE(frank): As we return immediately after this, we move the newValues
    // TODO(frank): make Values actually support move. Does not seem to happen now.
    this->state_ = currentState->decreaseLambda(params_, modelFidelity, std::move(newValues), newError);
    return true;
  } else if (!stopSearchingLambda) {  // we failed to solved the system or had no decrease in cost
    if (verbose)
      std::cout << "increasing lambda" << std::endl;
    auto* modifiedState = static_cast<internal::LevenbergMarquardtState*>(this->state_.get());
    modifiedState->increaseLambda(params_); // TODO(frank): make this functional with Values move

    // check if lambda is too big
    if (modifiedState->lambda >= params_.lambdaUpperBound) {
      if (params_.verbosity >= NonlinearOptimizerParams::TERMINATION ||
          params_.verbosityLM == LevenbergMarquardtParams::SUMMARY)
        std::cout << "Warning:  Levenberg-Marquardt giving up because "
                "cannot decrease error with maximum lambda" << std::endl;
      return true;
    } else {
      return false;  // only case where we will keep trying
    }
  } else {  // the change in the cost is very small and it is not worth trying bigger lambdas
    if (verbose)
      std::cout << "Levenberg-Marquardt: stopping as relative cost reduction is small" << std::endl;
    return true;
  }
}

// Implementation of iterate that uses custom tryLambda
template<typename PoseType>
GaussianFactorGraph::shared_ptr WNOALevenbergMarquardtOptimizer<PoseType>::iterate() {
  auto currentState = static_cast<const State*>(this->state_.get());

  gttic(LM_iterate);

  // Linearize graph - this will call our custom linearize() method
  if (params_.verbosityLM >= LevenbergMarquardtParams::DAMPED)
    std::cout << "linearizing = " << std::endl;
  GaussianFactorGraph::shared_ptr linear = this->linearize();

  if(currentState->totalNumberInnerIterations==0) { // write initial error
    writeLogFile(currentState->error);

    if (params_.verbosityLM == LevenbergMarquardtParams::SUMMARY) {
      std::cout << "Initial error: " << currentState->error
           << ", values: " << currentState->values.size() << std::endl;
    }
  }

  // Only calculate diagonal of Hessian (expensive) once per outer iteration, if we need it
  VectorValues sqrtHessianDiagonal;
  if (params_.diagonalDamping) {
    sqrtHessianDiagonal = linear->hessianDiagonal();
    for (auto& [key, value] : sqrtHessianDiagonal) {
      value = value.cwiseMax(params_.minDiagonal).cwiseMin(params_.maxDiagonal).cwiseSqrt();
    }
  }

  // Keep increasing lambda until we make make progress - this calls our custom tryLambda
  while (!tryLambda(*linear, sqrtHessianDiagonal)) {
    auto newState = static_cast<const State*>(this->state_.get());
    writeLogFile(newState->error);
  }

  return linear;
}

// Helper method for logging
template<typename PoseType>
void WNOALevenbergMarquardtOptimizer<PoseType>::writeLogFile(double currentError) {
  auto currentState = static_cast<const State*>(this->state_.get());

  if (!params_.logFile.empty()) {
    std::ofstream os(params_.logFile.c_str(), std::ios::app);
    // use chrono to measure time in microseconds
    auto currentTime = std::chrono::high_resolution_clock::now();
    // Get the time spent in seconds and print it
    double timeSpent = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - this->startTime_).count() / 1e6;
    os << /*inner iterations*/ currentState->totalNumberInnerIterations << ","
        << timeSpent << ","
        << /*current error*/ currentError << "," << currentState->lambda << ","
        << /*outer iterations*/ currentState->iterations << std::endl;
  }
}

// Explicit template instantiation to ensure the template gets compiled
template class WNOALevenbergMarquardtOptimizer<Point1>;
template class WNOALevenbergMarquardtOptimizer<Point2>;
template class WNOALevenbergMarquardtOptimizer<Point3>;
template class WNOALevenbergMarquardtOptimizer<Pose2>;
template class WNOALevenbergMarquardtOptimizer<Pose3>;

} // namespace gtsam
