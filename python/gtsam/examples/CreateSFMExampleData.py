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
import numpy as np

import gtsam
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point2, Point3, Pose3, Rot3, SfmData, SfmTrack


def opengl_fixed_rotation() -> Rot3:
    """
    Return the fixed rotation for OpenGL convention.
    R = [ 1   0   0
          0  -1   0
          0   0  -1]
    """
    R_mat = np.array([[1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0],
                      [0.0, 0.0, -1.0]])
    return Rot3(R_mat)


def gtsam2opengl(pose_gtsam: Pose3) -> Pose3:
    """Convert GTSAM camera pose to OpenGL camera pose."""
    R = pose_gtsam.rotation()
    t = pose_gtsam.translation()
    R90 = opengl_fixed_rotation()
    cRw_openGL = R90.compose(R.inverse())
    t_openGL = cRw_openGL.rotate(Point3(-t[0], -t[1], -t[2]))
    return Pose3(cRw_openGL, t_openGL)


def write_bal_file(filename: str, data: SfmData) -> bool:
    """
    Write SfmData to BAL (Bundle Adjustment in the Large) format file.
    
    BAL format:
    - First line: num_cameras num_points num_observations
    - Following lines: camera_index point_index u v (observations)
    - Then: camera parameters (9 per camera: rodrigues[3], translation[3], f, k1, k2)
    - Finally: 3D point coordinates (3 per point: x, y, z)
    
    Args:
        filename: Path to output file
        data: SfmData object containing cameras, tracks, and measurements
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            num_cameras = data.numberCameras()
            num_points = data.numberTracks()
            
            # Count total observations
            num_observations = 0
            for j in range(num_points):
                track = data.track(j)
                num_observations += track.numberMeasurements()
            
            # Write header
            f.write(f"{num_cameras} {num_points} {num_observations}\n")
            
            # Write observations
            for j in range(num_points):
                track = data.track(j)
                for m_idx in range(track.numberMeasurements()):
                    camera_idx, measurement = track.measurement(m_idx)
                    # Get principal point from camera calibration
                    camera = data.camera(camera_idx)
                    u0 = camera.calibration().px()
                    v0 = camera.calibration().py()
                    
                    # Adjust pixel measurements: center of image is origin, flip Y
                    pixelBALx = measurement[0] - u0
                    pixelBALy = -(measurement[1] - v0)
                    
                    f.write(f"{camera_idx} {j} {pixelBALx:.20g} {pixelBALy:.20g}\n")
            
            f.write("\n")
            
            # Write camera parameters
            for i in range(num_cameras):
                camera = data.camera(i)
                pose_gtsam = camera.pose()
                calibration = camera.calibration()
                
                # Convert GTSAM pose to OpenGL pose
                pose_opengl = gtsam2opengl(pose_gtsam)
                
                # Convert rotation to Rodrigues vector
                rodrigues = gtsam.Rot3.Logmap(pose_opengl.rotation())
                
                # Get translation
                translation = pose_opengl.translation()
                
                # Get calibration parameters (f, k1, k2)
                f_val = calibration.fx()  # focal length
                k1 = calibration.k1()
                k2 = calibration.k2()
                
                # Write each parameter on a separate line (matching C++ format)
                # Use high precision to match C++ output (precision 20)
                f.write(f"{rodrigues[0]:.20g}\n")
                f.write(f"{rodrigues[1]:.20g}\n")
                f.write(f"{rodrigues[2]:.20g}\n")
                f.write(f"{translation[0]:.20g}\n")
                f.write(f"{translation[1]:.20g}\n")
                f.write(f"{translation[2]:.20g}\n")
                f.write(f"{f_val:.20g}\n")
                f.write(f"{k1:.20g}\n")
                f.write(f"{k2:.20g}\n")
                f.write("\n")
            
            # Write 3D points (each coordinate on separate line)
            for j in range(num_points):
                track = data.track(j)
                point = track.point3()
                f.write(f"{point[0]:.20g}\n")
                f.write(f"{point[1]:.20g}\n")
                f.write(f"{point[2]:.20g}\n")
                f.write("\n")
        
        return True
    except Exception as e:
        print(f"Error in write_bal_file: {e}")
        return False


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
    
    # Write to BAL format
    if not write_bal_file(filename, data):
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

