import numpy as np
import numpy.linalg as npla
from math import cos, sin, factorial
from scipy.sparse import bmat
from scipy.sparse.linalg import inv
import pyvista as pv
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Any

import gtsam


# Plot the SE(3) transform
def plot_se3(plotter, T, scale=0.05, color=None, opacity=0.3):
    """
    Plot the SE(3) transform T using PyVista Arrows.
    """
    # Translation
    origin = T[:3, 3]
    
    # Axes directions (scaled)
    x_axis = T[:3, 0] * scale
    y_axis = T[:3, 1] * scale
    z_axis = T[:3, 2] * scale
    
    # Create and add arrow meshes
    arrow_x = pv.Arrow(start=origin, direction=x_axis, scale=scale, shaft_radius=0.2*scale)
    arrow_y = pv.Arrow(start=origin, direction=y_axis, scale=scale, shaft_radius=0.2*scale)
    arrow_z = pv.Arrow(start=origin, direction=z_axis, scale=scale, shaft_radius=0.2*scale)

    if color is None:
        plotter.add_mesh(arrow_x, color='red', opacity=opacity)
        plotter.add_mesh(arrow_y, color='green', opacity=opacity)
        plotter.add_mesh(arrow_z, color='blue', opacity=opacity)
    else:
        plotter.add_mesh(arrow_x, color=color, opacity=opacity)
        plotter.add_mesh(arrow_y, color=color, opacity=opacity)
        plotter.add_mesh(arrow_z, color=color, opacity=opacity)

# Plot the covariance
def plot_covariance(plotter, T, P, nstd=1, color='lightgray', opacity=0.2):
    """
    Plot the covariance ellipsoid at the given SE(3) transform T.
    """
    # Extract the rotation and translation from T
    R = T[:3, :3]
    t = T[:3, 3]

    # Do an eigen decomposition of the covariance matrix
    D, V = npla.eigh(R @ P @ R.T)
    W = nstd * V @ np.diag(np.sqrt(D))
    
    # Create an ellipsoid mesh
    ellipsoid = pv.Sphere(radius=1, theta_resolution=50, phi_resolution=50)
    
    # Scale the ellipsoid by the covariance matrix and translate it into place
    ellipsoid.points = (W @ ellipsoid.points.T).T + t
    
    plotter.add_mesh(ellipsoid, color=color, opacity=opacity)

# Make the axes equal
def axes_equal(ax):
    # Ensure equal scaling for all axes
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    # Find the maximum range
    max_range = max(
        abs(x_limits[1] - x_limits[0]),
        abs(y_limits[1] - y_limits[0]),
        abs(z_limits[1] - z_limits[0])
    ) / 2.0

    # Calculate the midpoints
    x_mid = (x_limits[0] + x_limits[1]) / 2.0
    y_mid = (y_limits[0] + y_limits[1]) / 2.0
    z_mid = (z_limits[0] + z_limits[1]) / 2.0

    # Set the new limits
    ax.set_xlim([x_mid - max_range, x_mid + max_range])
    ax.set_ylim([y_mid - max_range, y_mid + max_range])
    ax.set_zlim([z_mid - max_range, z_mid + max_range])

# Define a function to save the image to high-res PNG
# and crop it to the bounding box of the non-transparent part
def save_image(plotter):
    # Export as before
    plotter.export_gltf('output.gltf', True, True, False)
    
    # Take screenshot with transparent background
    temp_file = 'output_temp.png'
    plotter.screenshot(temp_file, transparent_background=True, window_size=[15360, 8640])
    
    # Now crop the image using PIL
    from PIL import Image

    # Disable decompression bomb warning
    Image.MAX_IMAGE_PIXELS = None
    
    # Open the image
    img = Image.open(temp_file)
    
    # Get the alpha channel
    alpha = img.getchannel('A')
    
    # Get the bounding box of the non-transparent part
    bbox = alpha.getbbox()
    
    if bbox:
        # Crop to the bounding box
        cropped = img.crop(bbox)
        
        # Add a small margin (optional)
        margin = 20  # pixels
        width, height = cropped.size
        new_bbox = (
            max(0, bbox[0] - margin),
            max(0, bbox[1] - margin),
            min(img.width, bbox[2] + margin),
            min(img.height, bbox[3] + margin)
        )
        cropped_with_margin = img.crop(new_bbox)
        
        # Save the result
        cropped_with_margin.save('output.png')
        
        print(f"Image cropped from {img.size} to {cropped_with_margin.size}")
    else:
        # If for some reason no content was found, save the original
        img.save('output.png')
        print("No content found for cropping")
    
    # Remove the temporary file
    import os
    if os.path.exists(temp_file):
        os.remove(temp_file)

def plot_poses(plotter: pv.Plotter,
               poses: List[gtsam.Pose3],
               covariances: List[np.ndarray] | None = None,
               nstd = 3.762,
               color = 'blue',
               scale = 0.15,
               opacity = 1.0):
    # Plot the posterior trajectory and covariances
    for k, pose in enumerate(poses):
        pose_inv = pose.inverse().matrix()
        plot_se3(plotter, pose_inv, scale=0.15, color=color, opacity=opacity)
        
        if covariances is not None:
            plot_covariance(plotter, pose_inv, covariances[k], nstd=nstd, color=color)
            