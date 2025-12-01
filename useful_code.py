
import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_rotation_matrix(quaternion):
    x, y, z, w = quaternion
    
    # Calculate the elements of the rotation matrix
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    
    return R

def plot_link_coordinate_frames(robot_id, link_indices, axis_length=0.1, duration=0):
    """
    Plots the coordinate frames of the specified links of a robot in PyBullet.

    Parameters:
    - robot_id: The ID of the robot in PyBullet.
    - link_indices: A list of link indices for which to plot the coordinate frames.
    - axis_length: The length of the axes to draw (default is 0.1).
    - duration: How long the lines should remain visible (0 means permanent).
    """
    for link_index in link_indices:
        # Get the position and orientation of the link in world coordinates
        link_state = p.getLinkState(robot_id, link_index)
        link_pos = link_state[4]  # World position of the link (x, y, z)
        link_orn = link_state[5]  # World orientation of the link (quaternion)

        # Convert the quaternion to a rotation matrix
        rot_matrix = R.from_quat(link_orn).as_matrix()

        # Define the local axes in the link frame
        x_axis = np.array([axis_length, 0, 0])  # x-axis in the local frame
        y_axis = np.array([0, axis_length, 0])  # y-axis in the local frame
        z_axis = np.array([0, 0, axis_length])  # z-axis in the local frame

        # Rotate the local axes to the world frame
        x_axis_world = rot_matrix @ x_axis
        y_axis_world = rot_matrix @ y_axis
        z_axis_world = rot_matrix @ z_axis

        # Add the axes as debug lines
        p.addUserDebugLine(link_pos, link_pos + x_axis_world, [1, 0, 0], lineWidth=2, lifeTime=duration)  # Red for x-axis
        p.addUserDebugLine(link_pos, link_pos + y_axis_world, [0, 1, 0], lineWidth=2, lifeTime=duration)  # Green for y-axis
        p.addUserDebugLine(link_pos, link_pos + z_axis_world, [0, 0, 1], lineWidth=2, lifeTime=duration)  # Blue for z-axis


