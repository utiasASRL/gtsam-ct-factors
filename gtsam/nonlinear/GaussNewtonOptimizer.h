/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    GaussNewtonOptimizer.h
 * @brief
 * @author  Richard Roberts
 * @date   Feb 26, 2012
 */

#pragma once

#include <gtsam/nonlinear/NonlinearOptimizer.h>

#include <type_traits>

namespace gtsam {

class GaussNewtonOptimizer;

/** Parameters for Gauss-Newton optimization, inherits from
 * NonlinearOptimizationParams.
 */
class GTSAM_EXPORT GaussNewtonParams : public NonlinearOptimizerParams {
public:
  using OptimizerType = GaussNewtonOptimizer;
};

/**
 * This class performs Gauss-Newton nonlinear optimization
 */
class GTSAM_EXPORT GaussNewtonOptimizer : public NonlinearOptimizer {

protected:
  GaussNewtonParams params_;

public:
  /// @name Standard interface
  /// @{

  /** Standard constructor, requires a nonlinear factor graph, initial
   * variable assignments, and optimization parameters.  For convenience this
   * version takes plain objects instead of shared pointers, but internally
   * copies the objects.
   * @param graph The nonlinear factor graph to optimize
   * @param initialValues The initial variable assignments
   * @param params The optimization parameters
   */
 GaussNewtonOptimizer(const NonlinearFactorGraph& graph, const Values& initialValues,
                      const GaussNewtonParams& params = GaussNewtonParams());

  /** Constructor that preserves derived graph behavior while copying the graph. */
  template <typename GraphType,
            typename = std::enable_if_t<
                std::is_base_of_v<NonlinearFactorGraph, GraphType> &&
                !std::is_same_v<NonlinearFactorGraph, GraphType>>>
  GaussNewtonOptimizer(const GraphType& graph, const Values& initialValues,
                       const GaussNewtonParams& params = GaussNewtonParams())
      : GaussNewtonOptimizer(std::make_shared<GraphType>(graph), initialValues,
                             params) {}

  /** Standard constructor, requires a nonlinear factor graph, initial
   * variable assignments, and optimization parameters.  For convenience this
   * version takes plain objects instead of shared pointers, but internally
   * copies the objects.
   * @param graph The nonlinear factor graph to optimize
   * @param initialValues The initial variable assignments
   */
  GaussNewtonOptimizer(const NonlinearFactorGraph& graph, const Values& initialValues,
                       const Ordering& ordering);

  /** Constructor with ordering that preserves derived graph behavior. */
  template <typename GraphType,
            typename = std::enable_if_t<
                std::is_base_of_v<NonlinearFactorGraph, GraphType> &&
                !std::is_same_v<NonlinearFactorGraph, GraphType>>>
  GaussNewtonOptimizer(const GraphType& graph, const Values& initialValues,
                       const Ordering& ordering)
      : GaussNewtonOptimizer(std::make_shared<GraphType>(graph), initialValues,
                             ordering) {}
  /// @}

  /// @name Advanced interface
  /// @{

  /** Virtual destructor */
  ~GaussNewtonOptimizer() override {}

  /** 
   * Perform a single iteration, returning GaussianFactorGraph corresponding to 
   * the linearized factor graph.
   */
  GaussianFactorGraph::shared_ptr iterate() override;

  /** Read-only access the parameters */
  const GaussNewtonParams& params() const { return params_; }

  /// @}

protected:
  /** Access the parameters (base class version) */
  const NonlinearOptimizerParams& _params() const override { return params_; }

  /** Internal function for computing a COLAMD ordering if no ordering is specified */
  GaussNewtonParams ensureHasOrdering(GaussNewtonParams params, const NonlinearFactorGraph& graph) const;

 private:
  GaussNewtonOptimizer(std::shared_ptr<const NonlinearFactorGraph> graph,
                       const Values& initialValues,
                       const GaussNewtonParams& params);

  GaussNewtonOptimizer(std::shared_ptr<const NonlinearFactorGraph> graph,
                       const Values& initialValues, const Ordering& ordering);

};

}
