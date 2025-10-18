"""
GTSAM Copyright 2010, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved
Authors: Frank Dellaert, et al. (see THANKS for the full author list)

See LICENSE for the license information

Create some example data that for inclusion in the data folder

Author: Frank Dellaert
Date: Oct 18, 2025
"""

import math
import os

import gtsam
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point3, Pose3, Rot3, SfmData, SfmTrack

def create_example_bal_file(filename: str, points: list[Point3],
                            pose1: Pose3, pose2: Pose3,
                            K: Cal3Bundler = Cal3Bundler()) -> None:
    """
    Create an example BAL file with two cameras observing multiple 3D points.
    
    Args:
        filename: Output file path
        points: List of 3D points to observe
        pose1: First camera pose
        pose2: Second camera pose
        K: Camera calibration (default: identity Cal3Bundler)
    """
    # Create SfmData to hold all data
    data = SfmData()
    
    # Create two cameras and add them to data
    data.addCamera(PinholeCameraCal3Bundler(pose1, K))
    data.addCamera(PinholeCameraCal3Bundler(pose2, K))
    
    for p in points:
        # Create the track
        track = SfmTrack(p)
        # Set RGB color to white
        track.r = 1.0
        track.g = 1.0
        track.b = 1.0
        
        # Project points in both cameras
        camera1 = PinholeCameraCal3Bundler(pose1, K)
        camera2 = PinholeCameraCal3Bundler(pose2, K)
        
        measurement1 = camera1.project(p)
        measurement2 = camera2.project(p)
        
        track.addMeasurement(0, measurement1)
        track.addMeasurement(1, measurement2)
        
        # Add track to data
        data.addTrack(track)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Write to BAL format using built-in function
    if not gtsam.writeBAL(filename, data):
        print(f"Error: could not write BAL file {filename}")


def create_5point_example1() -> None:
    """Create a simple 5-point example with small baseline."""
    # Create two camera poses
    aRb = Rot3.Ypr(math.pi / 2, 0, 0)
    aTb = Point3(0.1, 0, 0)
    pose1 = Pose3()
    pose2 = Pose3(aRb, aTb)
    
    # Create test data, we need at least 5 points
    points = [
        Point3(0, 0, 1),
        Point3(-0.1, 0, 1),
        Point3(0.1, 0, 1),
        Point3(0, 0.5, 0.5),
        Point3(0, -0.5, 0.5)
    ]
    
    # Assumes example is run from ${GTSAM_TOP} directory
    filename = "examples/Data/5pointExample1.txt"
    create_example_bal_file(filename, points, pose1, pose2)
    print(f"Created {filename}")


def create_5point_example2() -> None:
    """Create a 7-point example with larger baseline and calibration."""
    # Create two camera poses
    aRb = Rot3.Ypr(math.pi / 2, 0, 0)
    aTb = Point3(10, 0, 0)
    pose1 = Pose3()
    pose2 = Pose3(aRb, aTb)
    
    # Create test data, we need at least 5 points
    points = [
        Point3(0, 0, 100),
        Point3(-10, 0, 100),
        Point3(10, 0, 100),
        Point3(0, 50, 50),
        Point3(0, -50, 50),
        Point3(-20, 0, 80),
        Point3(20, -50, 80)
    ]
    
    # Assumes example is run from ${GTSAM_TOP} directory
    filename = "examples/Data/5pointExample2.txt"
    # Cal3Bundler(fx, k1, k2, u0, v0)
    K = Cal3Bundler(500, 0, 0, 0, 0)
    create_example_bal_file(filename, points, pose1, pose2, K)
    print(f"Created {filename}")


def create_18point_example1() -> None:
    """Create an 18-point example for more complex testing."""
    # Create two camera poses
    aRb = Rot3.Ypr(math.pi / 2, 0, 0)
    aTb = Point3(0.1, 0, 0)
    pose1 = Pose3()
    pose2 = Pose3(aRb, aTb)
    
    # Create test data with 18 points
    points = [
        Point3(0, 0, 1),         Point3(-0.1, 0, 1),      Point3(0.1, 0, 1),
        Point3(0, 0.5, 0.5),     Point3(0, -0.5, 0.5),    Point3(-1, -0.5, 2),
        Point3(-1, 0.5, 2),      Point3(0.25, -0.5, 1.5), Point3(0.25, 0.5, 1.5),
        Point3(-0.1, -0.5, 0.5), Point3(0.1, -0.5, 1),    Point3(0.1, 0.5, 1),
        Point3(-0.1, 0, 0.5),    Point3(-0.1, 0.5, 0.5),  Point3(0, 0, 0.5),
        Point3(0.1, -0.5, 0.5),  Point3(0.1, 0, 0.5),     Point3(0.1, 0.5, 0.5)
    ]
    
    # Assumes example is run from ${GTSAM_TOP} directory
    filename = "examples/Data/18pointExample1.txt"
    create_example_bal_file(filename, points, pose1, pose2)
    print(f"Created {filename}")


def main():
    """Create all example datasets."""
    print("Creating SFM example data files...")
    create_5point_example1()
    create_5point_example2()
    create_18point_example1()
    print("Done!")


if __name__ == "__main__":
    main()

