/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    WNOALevenbergMarquardtOptimizer.h
 * @brief   A specialized Levenberg-Marquardt optimizer for WNOAFactorGraph
 * @author  Sven Lilge
 * @date    Sep 9, 2025
 */

#pragma once

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/WNOAFactorGraph.h>

namespace gtsam {

/**
 * Custom Levenberg-Marquardt optimizer for WNOAFactorGraph
 */
template<typename PoseType>
class GTSAM_EXPORT WNOALevenbergMarquardtOptimizer: public LevenbergMarquardtOptimizer {
private:
  WNOAFactorGraph<PoseType>& wnoa_graph_;
  const LevenbergMarquardtParams params_;

public:
  /** Constructor specifically for WNOAFactorGraph 
   * @param graph The WNOA factor graph to optimize (must remain valid during optimization)
   * @param initialValues The initial variable assignments  
   * @param params The optimization parameters
   */
  WNOALevenbergMarquardtOptimizer(const WNOAFactorGraph<PoseType>& graph, 
                                  const Values& initialValues,
                                  const LevenbergMarquardtParams& params = LevenbergMarquardtParams());

  /** Constructor with ordering
   * @param graph The WNOA factor graph to optimize (must remain valid during optimization)
   * @param initialValues The initial variable assignments  
   * @param ordering The variable ordering
   * @param params The optimization parameters
   */
  WNOALevenbergMarquardtOptimizer(const WNOAFactorGraph<PoseType>& graph, 
                                  const Values& initialValues,
                                  const Ordering& ordering,
                                  const LevenbergMarquardtParams& params = LevenbergMarquardtParams());

  /** Override linearize to use WNOAFactorGraph's custom linearize method with cached interpolation data */
  GaussianFactorGraph::shared_ptr linearize() const override;

  /** Override iterate to use custom tryLambda and error calculation */
  GaussianFactorGraph::shared_ptr iterate() override;

  /** Custom tryLambda that uses WNOA graph for error calculation */
  bool tryLambda(const GaussianFactorGraph& linear, const VectorValues& sqrtHessianDiagonal);

  /** Write log file entry */
  void writeLogFile(double currentError);

protected:
  /** Access the parameters (override from base class) */
  const NonlinearOptimizerParams& _params() const override {
    return params_;
  }
};

} // namespace gtsam
