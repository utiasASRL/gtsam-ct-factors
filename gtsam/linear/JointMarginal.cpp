/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    JointMarginal.cpp
 * @brief   Block access to joint Gaussian covariance or information matrices
 * @author  Codex
 */

#include <gtsam/linear/JointMarginal.h>

#include <iostream>

namespace gtsam {

/* ************************************************************************* */
void JointMarginal::print(const std::string& s,
                          const KeyFormatter& formatter) const {
  std::cout << s << "Joint marginal on keys ";
  bool first = true;
  for (const auto& key : keys_) {
    if (!first)
      std::cout << ", ";
    else
      first = false;
    std::cout << formatter(key);
  }
  std::cout << ".  Use 'at' or 'operator()' to query matrix blocks."
            << std::endl;
}

}  // namespace gtsam
