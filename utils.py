
import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R

def check_edge_collision(robot_id, object_ids, joint_position_start, joint_position_end, discretization_step=0.01):
    """ 
    Checks for collision between two joint positions of a robot in PyBullet.
    Args:
        robot_id (int): The ID of the robot in PyBullet.
        object_ids (list): List of IDs of the objects in PyBullet.
        joint_position_start (list): List of joint positions to start from.
        joint_position_end (list): List of joint positions to get to.
        discretization_step (float): maximum interpolation distance before a new collision check is performed.
    Returns:
        bool: True if a collision is detected, False otherwise.
    """
    # Early exit
    start = np.array(joint_position_start)
    end = np.array(joint_position_end)
    if np.allclose(start, end):
        return check_node_collision(robot_id, object_ids, start.tolist())

    delta = np.abs(end - start)
    #need maximum 50 steps to get 0.01 discretization
    steps = int(np.round(np.max(delta) / discretization_step))
    for alpha in np.linspace(0.0, 1.0, steps + 10):
        q = (start + alpha * (end - start)).tolist()
        if check_node_collision(robot_id, object_ids, q):
            return True
    return False

def check_node_collision(robot_id, object_ids, joint_position):
    """
    Checks for collisions between a robot and an object in PyBullet. 

    Args:
        robot_id (int): The ID of the robot in PyBullet.
        object_id (int): The ID of the object in PyBullet.
        joint_position (list): List of joint positions. 

    Returns:
        bool: True if a collision is detected, False otherwise.
    """
    # set joint positions
    for joint_index, joint_pos in enumerate(joint_position):
        p.resetJointState(robot_id, joint_index, joint_pos)

    # Perform collision check for all links
    for object_id in object_ids:    # Check for each object
        for link_index in range(0, p.getNumJoints(robot_id)): # Check for each link of the robot
            contact_points = p.getClosestPoints(
                bodyA=robot_id, bodyB=object_id, distance=0.01, linkIndexA=link_index
            )
            if contact_points:  # If any contact points exist, a collision is detected
                return True # exit early
    return False
def plot_rrt_edge(robot_id, q_from, q_to, ee_link_index=2,
                  line_color=[0, 1, 0], line_width=1, duration=0):
    """
    Draw a single edge of the RRT (in workspace) between two joint configs.
    """
    p_from = get_ee_position(robot_id, q_from, ee_link_index)
    p_to   = get_ee_position(robot_id, q_to, ee_link_index)

    p.addUserDebugLine(
        lineFromXYZ=p_from.tolist(),
        lineToXYZ=p_to.tolist(),
        lineColorRGB=line_color,
        lineWidth=line_width,
        lifeTime=duration,   # 0 => persistent
    )

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


def get_ee_position(robot_id, joint_angles, ee_link_index):
    # Set joint states
    for j, q in enumerate(joint_angles):
        p.resetJointState(robot_id, j, float(q))

    # Get full link state to obtain frame pose
    link_state = p.getLinkState(robot_id, ee_link_index, computeForwardKinematics=1)

    # worldLinkFramePosition & worldLinkFrameOrientation
    link_pos = np.array(link_state[4], dtype=float)
    link_orn = link_state[5]

    # Rotation matrix from quaternion
    rot = np.array(quaternion_to_rotation_matrix(link_orn), dtype=float)

    # Tip offset in link frame (adjust if your geometry changes)
    tip_offset_local = np.array([0.475, 0.0, 0.0], dtype=float)

    # World position of end-effector
    ee_pos = link_pos + rot @ tip_offset_local
    return ee_pos

def plot_path(robot_id, path, line_color=[0, 1, 1], line_width=2, duration=0):
    """
    Plots a path defined by joint configurations in PyBullet.

    Parameters:
    - robot_id: The ID of the robot in PyBullet.
    - path: Iterable of joint configurations (iterable of length num_joints).
    - line_color: RGB list describing the color of the debug line.
    - line_width: Width of the debug line.
    - duration: Lifetime of the debug line (0 keeps it indefinitely).
    """
    configs = np.asarray(path, dtype=float)
    if configs.ndim == 1:
        configs = configs[np.newaxis, :]

    if len(configs) < 2:
        return []

    points = [np.array(get_ee_position(robot_id, cfg, ee_link_index=2), dtype=float) for cfg in configs]
    
    line_ids = []
    for start, end in zip(points[:-1], points[1:]):
        line_id = p.addUserDebugLine(
            lineFromXYZ=start.tolist(),
            lineToXYZ=end.tolist(),
            lineColorRGB=line_color,
            lineWidth=line_width,
            lifeTime=duration,
        )
        line_ids.append(line_id)

    return line_ids