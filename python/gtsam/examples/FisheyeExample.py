"""
GTSAM Copyright 2010, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved
Authors: Frank Dellaert, et al. (see THANKS for the full author list)

See LICENSE for the license information

A visualSLAM example for the structure-from-motion problem on a
simulated dataset. This version uses a fisheye camera model and a GaussNewton
solver to solve the graph in one batch

Author: Rohan Bansal
Date: Oct 17, 2025
"""

"""
A structure-from-motion example with landmarks
 - The landmarks form a 10 meter cube
 - The robot rotates around the landmarks, always facing towards the cube
"""

import numpy as np

import gtsam
from gtsam import symbol_shorthand

L = symbol_shorthand.L
X = symbol_shorthand.X

from gtsam.examples import SFMdata

from gtsam import (Cal3Fisheye, GaussNewtonOptimizer, GaussNewtonParams,
                   GenericProjectionFactorCal3Fisheye,
                   NonlinearFactorGraph, PinholeCameraCal3Fisheye,
                   Point3, Pose3, PriorFactorPoint3, PriorFactorPose3,
                   Rot3, Values)


def main():
    """
    A structure-from-motion example using a fisheye camera model.
    
    Camera observations of landmarks (i.e. pixel coordinates) will be stored as Point2 (x, y).

    Each variable in the system (poses and landmarks) must be identified with a unique key.
    We can either use simple integer keys (1, 2, 3, ...) or symbols (X1, X2, L1).
    Here we will use Symbols

    In GTSAM, measurement functions are represented as 'factors'. Several common factors
    have been provided with the library for solving robotics/SLAM/Bundle Adjustment problems.
    Here we will use Projection factors to model the camera's landmark observations.
    Also, we will initialize the robot at some location using a Prior factor.

    This example uses a fisheye camera model (Cal3Fisheye) with distortion parameters.
    """

    # Define the camera calibration parameters
    K = Cal3Fisheye(
        278.66, 278.48, 0.0, 319.75, 241.96,
        -0.013721808247486035, 0.020727425669427896,
        -0.012786476702685545, 0.0025242267320687625
    )

    # Define the camera observation noise model, 1 pixel stddev
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

    # Create the set of ground-truth landmarks
    points = SFMdata.createPoints()

    # Create the set of ground-truth poses
    poses = SFMdata.createPoses()

    # Create a Factor Graph and Values to hold the new data
    graph = NonlinearFactorGraph()
    initial_estimate = Values()

    # Add a prior on pose x0, 0.1 rad on roll,pitch,yaw, and 30cm std on x,y,z
    pose_prior = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])
    )
    graph.add(PriorFactorPose3(X(0), poses[0], pose_prior))

    # Add a prior on landmark l0
    point_prior = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    graph.add(PriorFactorPoint3(L(0), points[0], point_prior))

    # Add initial guesses to all observed landmarks
    # Intentionally initialize the variables off from the ground truth
    delta_point = Point3(-0.25, 0.20, 0.15)
    for j, point in enumerate(points):
        initial_estimate.insert(L(j), point + delta_point)

    # Loop over the poses, adding the observations to the graph
    for i, pose in enumerate(poses):
        # Add factors for each landmark observation
        for j, point in enumerate(points):
            camera = PinholeCameraCal3Fisheye(pose, K)
            measurement = camera.project(point)
            factor = GenericProjectionFactorCal3Fisheye(
                measurement, measurement_noise, X(i), L(j), K
            )
            graph.add(factor)

        # Add an initial guess for the current pose
        # Intentionally initialize the variables off from the ground truth
        delta_pose = Pose3(
            Rot3.Rodrigues(-0.1, 0.2, 0.25),
            Point3(0.05, -0.10, 0.20)
        )
        initial_estimate.insert(X(i), pose.compose(delta_pose))

    # Configure and run the optimizer
    params = GaussNewtonParams()
    params.setVerbosity("TERMINATION")
    params.setMaxIterations(10000)

    print("Optimizing the factor graph")
    optimizer = GaussNewtonOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    print("Optimization complete")

    print(f"initial error={graph.error(initial_estimate)}")
    print(f"final error={graph.error(result)}")

    # Save the factor graph visualization
    try:
        graph.saveGraph("python/gtsam/examples/fisheye_batch.dot", result)
        print("Saved graph to python/gtsam/examples/fisheye_batch.dot")
    except Exception as e:
        print(f"Could not save graph: {e}")


if __name__ == "__main__":
    main()

