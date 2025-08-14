/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    ContinuousTimeParams.cpp
 * @brief   Parameters for continuous-time optimization
 * @date August 13, 2025
 * @author Sven Lilge
 */

#include <gtsam/nonlinear/ContinuousTimeParams.h>
#include <iostream>
#include <string>

using namespace std;

namespace gtsam {


/* ************************************************************************* */
void ContinuousTimeParams::print(const std::string& str) const {
  NonlinearOptimizerParams::print(str);
  std::cout << "Number of interpolated states: " << interpolatedStates.size() << "\n";
  std::cout << "Delta threshold: " << deltaThreshold << "\n";
  std::cout << "Max inner iterations: " << maxInnerIterations << "\n";
  std::cout.flush();
}

} /* namespace gtsam */

