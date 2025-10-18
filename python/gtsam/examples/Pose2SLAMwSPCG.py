"""
GTSAM Copyright 2010, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved
Authors: Frank Dellaert, et al. (see THANKS for the full author list)

See LICENSE for the license information

A 2D Pose SLAM example using the SubgraphSolver (SPCG).
Author: Yong-Dian Jian (C++), Translated to Python
Date: June 2, 2012
"""

import math
import numpy as np
import gtsam


def main():
    """Main function demonstrating 2D Pose SLAM with SubgraphSolver (SPCG)."""
    
    # 1. Create a factor graph container and add factors to it
    graph = gtsam.NonlinearFactorGraph()
    
    # 2a. Add a prior on the first pose, setting it to the origin
    prior = gtsam.Pose2(0.0, 0.0, 0.0)  # prior at origin
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
    graph.add(gtsam.PriorFactorPose2(1, prior, prior_noise))
    
    # 2b. Add odometry factors
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    graph.add(gtsam.BetweenFactorPose2(1, 2, gtsam.Pose2(2.0, 0.0, math.pi/2), odometry_noise))
    graph.add(gtsam.BetweenFactorPose2(2, 3, gtsam.Pose2(2.0, 0.0, math.pi/2), odometry_noise))
    graph.add(gtsam.BetweenFactorPose2(3, 4, gtsam.Pose2(2.0, 0.0, math.pi/2), odometry_noise))
    graph.add(gtsam.BetweenFactorPose2(4, 5, gtsam.Pose2(2.0, 0.0, math.pi/2), odometry_noise))
    
    # 2c. Add the loop closure constraint
    model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    graph.add(gtsam.BetweenFactorPose2(5, 1, gtsam.Pose2(0.0, 0.0, 0.0), model))
    
    print("\nFactor Graph:")
    print(graph)
    
    # 3. Create the data structure to hold the initial estimate to the solution
    initial_estimate = gtsam.Values()
    initial_estimate.insert(1, gtsam.Pose2(0.5, 0.0, 0.2))
    initial_estimate.insert(2, gtsam.Pose2(2.3, 0.1, 1.1))
    initial_estimate.insert(3, gtsam.Pose2(2.1, 1.9, 2.8))
    initial_estimate.insert(4, gtsam.Pose2(-0.3, 2.5, 4.2))
    initial_estimate.insert(5, gtsam.Pose2(0.1, -0.7, 5.8))
    
    print("\nInitial Estimate:")
    print(initial_estimate)
    
    # 4. Single Step Optimization using Levenberg-Marquardt with SubgraphSolver
    parameters = gtsam.LevenbergMarquardtParams()
    parameters.setVerbosity("ERROR")
    parameters.setVerbosityLM("LAMBDA")
    
    # LM is still the outer optimization loop, but by specifying "ITERATIVE"
    # below we indicate that an iterative linear solver should be used. In
    # addition, the *type* of the iterativeParams decides on the type of
    # iterative solver, in this case the SPCG (subgraph PCG)
    parameters.setLinearSolverType("ITERATIVE")
    parameters.setIterativeParams(gtsam.SubgraphSolverParameters())
    
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, parameters)
    result = optimizer.optimize()
    
    print("\nFinal Result:")
    print(result)
    print(f"subgraph solver final error = {graph.error(result)}")
    
    return 0


if __name__ == "__main__":
    exit(main())
