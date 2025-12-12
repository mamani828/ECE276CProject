import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R


def check_edge_collision(
    robot_id,
    object_ids,
    joint_position_start,
    joint_position_end,
    discretization_step=0.01,
):
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
    # need maximum 50 steps to get 0.01 discretization
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
    for object_id in object_ids:  # Check for each object
        for link_index in range(
            0, p.getNumJoints(robot_id)
        ):  # Check for each link of the robot
            contact_points = p.getClosestPoints(
                bodyA=robot_id, bodyB=object_id, distance=0.01, linkIndexA=link_index
            )
            if contact_points:  # If any contact points exist, a collision is detected
                return True  # exit early
    return False


def plot_rrt_edge(
    robot_id,
    q_from,
    q_to,
    ee_link_index=2,
    line_color=[0, 1, 0],
    line_width=1,
    duration=0,
):
    """
    Draw a single edge of the RRT (in workspace) between two joint configs.
    """
    p_from = get_ee_position(robot_id, q_from, ee_link_index)
    p_to = get_ee_position(robot_id, q_to, ee_link_index)

    p.addUserDebugLine(
        lineFromXYZ=p_from.tolist(),
        lineToXYZ=p_to.tolist(),
        lineColorRGB=line_color,
        lineWidth=line_width,
        lifeTime=duration,  # 0 => persistent
    )


def quaternion_to_rotation_matrix(quaternion):
    x, y, z, w = quaternion

    # Calculate the elements of the rotation matrix
    R = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )

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
        p.addUserDebugLine(
            link_pos, link_pos + x_axis_world, [1, 0, 0], lineWidth=2, lifeTime=duration
        )  # Red for x-axis
        p.addUserDebugLine(
            link_pos, link_pos + y_axis_world, [0, 1, 0], lineWidth=2, lifeTime=duration
        )  # Green for y-axis
        p.addUserDebugLine(
            link_pos, link_pos + z_axis_world, [0, 0, 1], lineWidth=2, lifeTime=duration
        )  # Blue for z-axis


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

    points = [
        np.array(get_ee_position(robot_id, cfg, ee_link_index=2), dtype=float)
        for cfg in configs
    ]

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


def is_state_valid(robot_id, joint_indices, q_target, obstacle_ids, check_self=True):
    """
    Checks if a target configuration q_target is valid.

    Args:
        robot_id: PyBullet body ID of the robot.
        joint_indices: List of joint indices corresponding to q_target.
        q_target: List of joint angles (target configuration).
        obstacle_ids: List of PyBullet body IDs for environmental obstacles.
        check_self: Boolean, whether to check for robot-robot collisions.

    Returns:
        True if valid (no collisions), False otherwise.
    """
    # 1. Save current state so we can restore it later
    # (We don't want to mess up the simulation just to check a hypothetical state)
    current_states = []
    for j_idx in joint_indices:
        current_states.append(p.getJointState(robot_id, j_idx)[0])

    # 2. Teleport robot to the target configuration
    for i, j_idx in enumerate(joint_indices):
        p.resetJointState(robot_id, j_idx, q_target[i])

    # 3. Step simulation "forward" by 0 seconds to update contact points
    # (PyBullet requires this to update the AABBs for collision detection)
    p.performCollisionDetection()

    is_valid = True

    # 4. Check Environment Collision (Robot vs Obstacles)
    for obs_id in obstacle_ids:
        # getContactPoints returns a list. If not empty, there is a collision.
        contacts = p.getContactPoints(bodyA=robot_id, bodyB=obs_id)
        if len(contacts) > 0:
            is_valid = False
            print(f"Collision detected with obstacle ID {obs_id}")
            break

    # 5. Check Self-Collision (Robot vs Robot)
    if is_valid and check_self:
        # getContactPoints with only bodyA checks for self-collision
        self_contacts = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)

        # Filter out adjacent links (they are connected by joints and always "touch")
        # We only care if non-adjacent links touch (e.g., Left Hand hits Right Shoulder)
        real_self_collision = False
        for contact in self_contacts:
            link_a = contact[3]
            link_b = contact[4]
            # Check if links are not adjacent (indices differ by more than 1)
            # Note: This is a simple heuristic. For complex trees, use a specific adjacency matrix.
            # But for your tree (Left chain vs Right chain), indices will differ significantly.
            if abs(link_a - link_b) > 1:
                real_self_collision = True
                break

        if real_self_collision:
            print(f"Self-collision detected between link {link_a} and {link_b}")
            is_valid = False

    # 6. Restore original state
    for i, j_idx in enumerate(joint_indices):
        p.resetJointState(robot_id, j_idx, current_states[i])

    return is_valid


def mark_goal_configurations_dual(
    robot_id, movable_joints, goal_configs, left_ee_idx, right_ee_idx
):
    """
    Takes GOAL CONFIGURATIONS (Joint Angles for BOTH arms), calculates where the EEs will be,
    and places static visual markers there.

    Args:
        robot_id (int): Body ID.
        movable_joints (list[int]): Flat list of joint indices for BOTH arms.
        goal_configs (list[list[float]]): List of configurations (each config must match len(movable_joints)).
        left_ee_idx (int): Joint index for Left EE tip.
        right_ee_idx (int): Joint index for Right EE tip.
    """
    # Colors: Cyan for Left, Magenta for Right, Black for text
    c_left = [0, 1, 1, 0.6]
    c_right = [1, 0, 1, 0.6]
    c_text = [0, 0, 0]

    print(f"\n--- Visualizing {len(goal_configs)} Goal Configurations ---")

    # Save current state to restore later
    current_states = [p.getJointState(robot_id, j)[0] for j in movable_joints]

    marker_ids = []

    for i, q_goal in enumerate(goal_configs):
        # 1. Teleport robot to this goal configuration
        # q_goal must be a flat list covering all movable_joints (left + right)
        for k, j_idx in enumerate(movable_joints):
            p.resetJointState(robot_id, j_idx, q_goal[k])

        # 2. Get EE positions with computeForwardKinematics=True
        # We need this because we haven't stepped the simulation
        state_left = p.getLinkState(
            robot_id, left_ee_idx, computeForwardKinematics=True
        )
        state_right = p.getLinkState(
            robot_id, right_ee_idx, computeForwardKinematics=True
        )

        # Use index [4] for the Link Frame (visual mesh origin)
        pos_left = state_left[4]
        pos_right = state_right[4]

        # 3. Create Markers
        # Left Marker (Cyan)
        v_l = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=c_left)
        id_l = p.createMultiBody(baseVisualShapeIndex=v_l, basePosition=pos_left)
        marker_ids.append(id_l)

        # Right Marker (Magenta)
        v_r = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=c_right)
        id_r = p.createMultiBody(baseVisualShapeIndex=v_r, basePosition=pos_right)
        marker_ids.append(id_r)

        # 4. Add Labels
        p.addUserDebugText(
            f"L{i}", [pos_left[0], pos_left[1], pos_left[2] + 0.1], c_text
        )
        p.addUserDebugText(
            f"R{i}", [pos_right[0], pos_right[1], pos_right[2] + 0.1], c_text
        )

        print(f"  Goal {i}: L={np.round(pos_left, 2)} R={np.round(pos_right, 2)}")

    # Restore original state
    for k, j_idx in enumerate(movable_joints):
        p.resetJointState(robot_id, j_idx, current_states[k])

    # Force visual update
    p.performCollisionDetection()

    return marker_ids


def mark_goal_configurations(robot_id, movable_joints, goal_configs, ee_idx):
    """
    Takes GOAL CONFIGURATIONS (Joint Angles), calculates where the EEs will be,
    and places static visual markers there.

    Args:
        robot_id (int): PyBullet Body ID of the robot.
        movable_joints (list[int]): Indices of the joints to control (e.g. [0, 1, 2]).
        goal_configs (list[list[float]]): List of target joint angles.
        ee_idx (int): The Link Index of the End Effector (Must be the FIXED joint/tip index).

    Returns:
        list[int]: A list of body IDs for the created markers (useful for cleanup).
    """
    # Colors: Cyan for EE, Black for Text
    c_ee = [0, 1, 1, 0.6]
    c_text = [0, 0, 0]

    print(f"\n--- Visualizing {len(goal_configs)} Goal Configurations ---")

    # Save current state to restore later
    current_states = [p.getJointState(robot_id, j)[0] for j in movable_joints]

    marker_ids = []

    for i, q_goal in enumerate(goal_configs):
        # 1. Teleport robot to this goal configuration
        for k, j_idx in enumerate(movable_joints):
            p.resetJointState(robot_id, j_idx, q_goal[k])

        # 2. Get EE position with computeForwardKinematics=True
        # We need this flag because we just teleported the joints without stepping simulation
        ee_state = p.getLinkState(robot_id, ee_idx, computeForwardKinematics=True)
        pos_ee = ee_state[4]  # Index 4 is the Link Frame position

        # 3. Create Marker (Sphere)
        v_ee = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=c_ee)
        marker_id = p.createMultiBody(baseVisualShapeIndex=v_ee, basePosition=pos_ee)
        marker_ids.append(marker_id)

        # 4. Create Label (Text)
        # Shift text slightly above the sphere so it's readable
        text_pos = [pos_ee[0], pos_ee[1], pos_ee[2] + 0.1]
        p.addUserDebugText(f"Goal {i}", text_pos, c_text, textSize=1.5)

        print(f"  Goal {i}: EE Position = {np.round(pos_ee, 2)}")

    # Restore original state so the robot snaps back to where it was before calling this
    for k, j_idx in enumerate(movable_joints):
        p.resetJointState(robot_id, j_idx, current_states[k])

    # Optional: Force a visual update of the robot mesh to the restored state
    p.performCollisionDetection()

    return marker_ids
