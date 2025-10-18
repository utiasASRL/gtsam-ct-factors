"""
GTSAM Copyright 2010, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved
Authors: Frank Dellaert, et al. (see THANKS for the full author list)

See LICENSE for the license information

Expressions version of Pose2SLAMExample.cpp
Author: Frank Dellaert (C++), Translated to Python
"""

import math
import numpy as np
import gtsam


def main():
    """Main function demonstrating Pose2 SLAM using standard factors.
    
    Note: The original C++ version uses ExpressionFactorGraph and Pose2_ expressions
    for automatic differentiation. Since these are not exposed in Python GTSAM,
    this translation uses the standard NonlinearFactorGraph with BetweenFactorPose2
    and PriorFactorPose2, which provide equivalent functionality.
    """
    
    # 1. Create a factor graph container and add factors to it
    # Note: Using NonlinearFactorGraph instead of ExpressionFactorGraph
    # since expressions are not exposed in Python GTSAM
    graph = gtsam.NonlinearFactorGraph()
    
    # 2a. Add a prior on the first pose, setting it to the origin
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
    graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(0, 0, 0), prior_noise))
    
    # For simplicity, we use the same noise model for odometry and loop closures
    model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    
    # 2b. Add odometry factors
    # These correspond to the between(x1, x2) expressions in the C++ version
    graph.add(gtsam.BetweenFactorPose2(1, 2, gtsam.Pose2(2, 0, 0), model))
    graph.add(gtsam.BetweenFactorPose2(2, 3, gtsam.Pose2(2, 0, math.pi/2), model))
    graph.add(gtsam.BetweenFactorPose2(3, 4, gtsam.Pose2(2, 0, math.pi/2), model))
    graph.add(gtsam.BetweenFactorPose2(4, 5, gtsam.Pose2(2, 0, math.pi/2), model))
    
    # 2c. Add the loop closure constraint
    graph.add(gtsam.BetweenFactorPose2(5, 2, gtsam.Pose2(2, 0, math.pi/2), model))
    
    print("\nFactor Graph:")
    print(graph)
    
    # 3. Create the data structure to hold the initial estimate to the solution
    # For illustrative purposes, these have been deliberately set to incorrect values
    initial_estimate = gtsam.Values()
    initial_estimate.insert(1, gtsam.Pose2(0.5, 0.0, 0.2))
    initial_estimate.insert(2, gtsam.Pose2(2.3, 0.1, -0.2))
    initial_estimate.insert(3, gtsam.Pose2(4.1, 0.1, math.pi/2))
    initial_estimate.insert(4, gtsam.Pose2(4.0, 2.0, math.pi))
    initial_estimate.insert(5, gtsam.Pose2(2.1, 2.1, -math.pi/2))
    
    print("\nInitial Estimate:")
    print(initial_estimate)
    
    # 4. Optimize the initial values using a Gauss-Newton nonlinear optimizer
    parameters = gtsam.GaussNewtonParams()
    parameters.setRelativeErrorTol(1e-5)
    parameters.setMaxIterations(100)
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, parameters)
    result = optimizer.optimize()
    
    print("\nFinal Result:")
    print(result)
    
    # 5. Calculate and print marginal covariances for all variables
    print("\nMarginal Covariances:")
    marginals = gtsam.Marginals(graph, result)
    
    for i in range(1, 6):
        covariance = marginals.marginalCovariance(i)
        print(f"x{i} covariance:")
        print(covariance)
        print()
    
    return 0


if __name__ == "__main__":
    exit(main())
