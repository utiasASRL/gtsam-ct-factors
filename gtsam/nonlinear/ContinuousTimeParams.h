/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file ContinuousTimeParams.h
 * @brief Parameters for continuous-time optimization
 * @date August 13, 2025
 * @author Sven Lilge
 */
#pragma once

#include <gtsam/nonlinear/NonlinearOptimizerParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

namespace gtsam {

class ContinuousTimeOptimizer;

/** Parameters for continuous-time optimization.  Note that this parameters
 * class inherits from NonlinearOptimizerParams, which specifies the parameters
 * common to all nonlinear optimization algorithms.  This class also contains
 * all of those parameters.
 */
class GTSAM_EXPORT ContinuousTimeParams: public NonlinearOptimizerParams {

public:

  using OptimizerType = ContinuousTimeOptimizer;

public:

  KeyVector interpolatedStates; ///< States that are considered "interpolated" (default: empty)
  double deltaThreshold = 1e-6; ///< Required threshold for the delta of the discrete states before the interpolated states are being updated
  int maxInnerIterations = 1; ///< Maximum number of inner iterations for the optimization (before updating the interpolated states)

  ContinuousTimeParams()
      : interpolatedStates() {
  }

  static void SetLegacyDefaults(ContinuousTimeParams* p) {
    // Relevant NonlinearOptimizerParams:
    p->maxIterations = 100;
    p->relativeErrorTol = 1e-5;
    p->absoluteErrorTol = 1e-5;
  }


  static ContinuousTimeParams LegacyDefaults() {
    ContinuousTimeParams p;
    SetLegacyDefaults(&p);
    return p;
  }

  ~ContinuousTimeParams() override {}
  void print(const std::string& str = "") const override;

  /// @name Getters/Setters, mainly for wrappers. Use fields above in C++.
  /// @{
  KeyVector getInterpolatedStates() const { return interpolatedStates; }
  void setInterpolatedStates(const KeyVector& states) { interpolatedStates = states; }
  double getDeltaThreshold() const { return deltaThreshold; }
  void setDeltaThreshold(double threshold) { deltaThreshold = threshold; }
  int getMaxInnerIterations() const { return maxInnerIterations; }
  void setMaxInnerIterations(int maxIter) { maxInnerIterations = maxIter; }
  // @}
  /// @name Clone
  /// @{

  /// @return a deep copy of this object
  std::shared_ptr<NonlinearOptimizerParams> clone() const {
    return std::shared_ptr<NonlinearOptimizerParams>(new ContinuousTimeParams(*this));
  }

  /// @}
};

}
