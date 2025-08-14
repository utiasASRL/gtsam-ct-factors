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


#pragma once

#include <gtsam/nonlinear/NonlinearOptimizer.h>
#include <gtsam/nonlinear/ContinuousTimeParams.h>
#include <gtsam/linear/VectorValues.h>
#include <chrono>

namespace gtsam {

/**
 * This class performs continuous-time optimization
 */
class GTSAM_EXPORT ContinuousTimeOptimizer: public NonlinearOptimizer {

protected:
  const ContinuousTimeParams params_; ///< CT parameters
  NonlinearFactorGraph remaining_graph_; ///< The graph with factors that do not involve interpolated keys

  KeyVector interpolatedKeys_; ///< Keys that are considered "interpolated" and will be eliminated
  KeyVector remainingKeys_; ///< Keys that will remain in the graph

  // startTime_ is a chrono time point
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime_; ///< time when optimization started

  void initTime();

public:
  typedef std::shared_ptr<ContinuousTimeOptimizer> shared_ptr;

  /// @name Constructors/Destructor
  /// @{

  /** Standard constructor, requires a nonlinear factor graph, initial
   * variable assignments, and optimization parameters.  For convenience this
   * version takes plain objects instead of shared pointers, but internally
   * copies the objects.
   * @param graph The nonlinear factor graph to optimize
   * @param initialValues The initial variable assignments
   * @param params The optimization parameters
   */
  ContinuousTimeOptimizer(const NonlinearFactorGraph& graph, const Values& initialValues,
                              const ContinuousTimeParams& params = ContinuousTimeParams());


  /** Virtual destructor */
  ~ContinuousTimeOptimizer() {
  }

  /// @}

  /// @name Standard interface
  /// @{

  /// print
  void print(const std::string& str = "") const {
    std::cout << str << "ContinuousTimeOptimizer" << std::endl;
    this->params_.print("  parameters:\n");
  }

  /// @}

  /// @name Advanced interface
  /// @{


  /** 
   * Perform a single iteration, returning GaussianFactorGraph corresponding to 
   * the linearized factor graph.
   */
  GaussianFactorGraph::shared_ptr iterate() override;

  /** Read-only access the parameters */
  const ContinuousTimeParams& params() const {
    return params_;
  }


  /** linearize, can be overwritten */
  virtual GaussianFactorGraph::shared_ptr linearize() const;


  /// @}

protected:

  /** Access the parameters (base class version) */
  const NonlinearOptimizerParams& _params() const override {
    return params_;
  }
};

}
