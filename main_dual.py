import random
import time
import numpy as np
import pybullet as p
import pybullet_data
from matplotlib import pyplot as plt

# Assuming these exist in your local directory as before
from rrt import RRT
from rrt_cbf import RRT_CBF
from useful_code import *
from sdf import make_pybullet_env_sdf, visualize_sdf_slice
from utils import is_state_valid

# Helper functions
def get_sphere_centers(robot_id, joint_q, spheres, movable_joints):
    """
    Given joint positions joint_q (length N),
    return list of world-frame sphere centers c_k(q) for 'spheres'.
    """
    # Set joint states in PyBullet
    # joint_q should correspond to the movable_joints list
    for i, joint_idx in enumerate(movable_joints):
        p.resetJointState(robot_id, joint_idx, float(joint_q[i]))

    centers = []
    for sph in spheres:
        link_idx = sph["link_index"]
        local_pos = np.asarray(sph["local_pos"], dtype=float)

        # Get link pose in world frame
        link_state = p.getLinkState(robot_id, link_idx, computeForwardKinematics=True)
        link_pos = np.asarray(link_state[0], dtype=float)
        link_orn = link_state[1]

        # Rotation matrix from quaternion
        R = np.array(p.getMatrixFromQuaternion(link_orn)).reshape(3, 3)

        # Transform local sphere center to world
        world_pos = link_pos + R @ local_pos
        centers.append(world_pos)

    return centers


def visualize_spherical_robot(robot_id, joint_q, spheres, movable_joints, color=[1, 0, 0, 0.4]):
    """
    Draw translucent spheres for the spherical approximation of the robot.
    """
    centers = get_sphere_centers(robot_id, joint_q, spheres, movable_joints)

    for center, sph in zip(centers, spheres):
        r = sph["radius"]

        visual_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=r,
            rgbaColor=color,  # [R,G,B,alpha]
        )

        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_id,
            basePosition=center.tolist(),
        )


def get_link_axis_and_length(robot_id, link_index, q_ref=None):
    """Compute local axis and length of link using a reference configuration."""
    # 1. Set reference configuration (zero pose)
    if q_ref is None:
        # Reset all joints to 0
        for j in range(p.getNumJoints(robot_id)):
            p.resetJointState(robot_id, j, 0.0)
            
    # 2. Get current link pose
    link_state = p.getLinkState(robot_id, link_index, computeForwardKinematics=True)
    pos_link = np.array(link_state[0])
    orn_link = link_state[1]
    R = np.array(p.getMatrixFromQuaternion(orn_link)).reshape(3, 3)

    # 3. Determine "Child" position to calculate length
    # In a tree, we look for a joint whose parent is THIS link.
    child_found = False
    pos_child = pos_link # Default if no child found

    # Scan all joints to find one whose parent is 'link_index'
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        parent_index = info[16] # Index 16 is parentIndex
        if parent_index == link_index:
            # We found a child link!
            child_state = p.getLinkState(robot_id, j, computeForwardKinematics=True)
            pos_child = np.array(child_state[0])
            child_found = True
            break # Just take the first child (assuming simple chain segments)

    # If no child joint (it's an end effector tip), make a synthetic length
    if not child_found:
        # Assuming cylinder aligned along X or similar, 
        # or just hardcode a small offset for the tip
        pos_child = pos_link + np.array([0.5, 0.0, 0.0]) # Matches URDF length roughly

    # 4. Calculate Direction and Length in World Frame
    dir_world = pos_child - pos_link
    length = np.linalg.norm(dir_world)
    
    if length < 1e-6:
        return np.array([1.0, 0.0, 0.0]), 0.0

    # 5. Transform to Local Frame: dir_local = R^T * dir_world
    dir_local = R.T @ (dir_world / length)
    return dir_local, length


def make_link_spheres_from_fk(
    robot_id, link_index, radius, q_ref=None, max_spacing_factor=1.5, min_spheres=2
):
    axis_local, length = get_link_axis_and_length(robot_id, link_index, q_ref=q_ref)

    # If the link is degenerate, just place one sphere
    if length < 1e-6:
        return [
            {
                "link_index": link_index,
                "local_pos": [0.0, 0.0, 0.0],
                "radius": radius,
            }
        ]

    # Decide max spacing between sphere centers based on radius
    max_spacing = max_spacing_factor * radius

    # Number of spheres needed to respect max spacing
    n_spheres = int(np.ceil(length / max_spacing)) + 1
    n_spheres = max(n_spheres, min_spheres)

    # Place spheres evenly from 0 to length along axis_local
    spheres = []
    for i in range(n_spheres):
        t = i / (n_spheres - 1)  # from 0 to 1
        local_pos = (axis_local * (length * t)).tolist()
        spheres.append(
            {
                "link_index": link_index,
                "local_pos": local_pos,
                "radius": radius,
            }
        )
    return spheres


# Main Code
if __name__ == "__main__":
    # Problem setup
    env = "simple"

    # Initialize PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Load the plane and NEW dual-arm robot
    ground_id = p.loadURDF("plane.urdf")
    # Ensure you are loading the updated URDF file name here
    arm_id = p.loadURDF("dual_three_link_robot.urdf", [0, 0, 0], useFixedBase=True)
    
    # Identify Movable Joints (Revolute Only)
    movable_joints = []
    movable_joint_names = []
    
    num_joints = p.getNumJoints(arm_id)
    print(f"Total joints in URDF: {num_joints}")
    
    for i in range(num_joints):
        info = p.getJointInfo(arm_id, i)
        j_type = info[2]
        j_name = info[1].decode('utf-8')
        # We only control Revolute joints (ignore Fixed joints at EE)
        if j_type == p.JOINT_REVOLUTE:
            movable_joints.append(i)
            movable_joint_names.append(j_name)
            
    print(f"Planning for joints: {movable_joints} ({movable_joint_names})")
    # This should be indices [0, 1, 2, 4, 5, 6] typically
    
    # Generate Collision Spheres for All Links
    q_ref = [0.0] * num_joints # Reference pose for calculation
    ROBOT_SPHERES = []
    
    # We generate spheres for every movable link. 
    # Note: movable_joints points to the joint, which usually has the same index as the child link.
    for link_idx in movable_joints:
        ROBOT_SPHERES += make_link_spheres_from_fk(
            arm_id,
            link_index=link_idx,
            radius=0.08, # Slightly smaller for dual arm to avoid self-collision at start
            q_ref=q_ref,
            max_spacing_factor=1.0,
            min_spheres=2,
        )

    # Setup Environment Obstacles
    collision_ids = [ground_id]
    
    if env == "simple":
        # collision_ids = []
        collision_positions = [
            [0.3, 0.5, 0.251],
            [-0.3, 0.3, 0.101],
            [-1, -0.15, 0.251],
            [-1, -0.15, 0.752],
            [-0.5, -1, 0.251],
            [0.5, -0.35, 0.201],
            [0.5, -0.35, 0.602],
        ]
        collision_orientations = [
            [0, 0, 0.5],
            [0, 0, 0.2],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0.25],
            [0, 0, 0.5],
        ]
        collision_scales = [0.5, 0.25, 0.5, 0.5, 0.5, 0.4, 0.4]

        # Having colorful cubes
        colors = [
        [1, 0, 0, 1],   # red
        [0, 1, 0, 1],   # green
        [0, 0, 1, 1],   # blue
        [1, 1, 0, 1],   # yellow
        [1, 0, 1, 1],   # magenta
        [0, 1, 1, 1],   # cyan
        [0.8, 0.4, 0, 1], # orange
        ]
    elif env == "complex":
        # collision_ids = []
        collision_positions = [
            [0.3, 0.5, 0.251],
            [-0.3, 0.3, 0.101],
            [-1, -0.15, 0.251],
            # [-1, -0.15, 0.752],
            [-0.5, -1, 0.251],
            # [-0.5, -1, 0.752],
            [0.5, -0.35, 0.201],
            [0.5, -0.35, 0.602],
            [-0.75, -0.575, 1.003], # Top cube
        ]
        collision_orientations = [
            [0, 0, 0.5],
            [0, 0, 0.2],
            [0, 0, 0],
            # [0, 0, 1],
            [0, 0, 0],
            # [0, 0, 0],
            [0, 0, 0.25],
            [0, 0, 0.5],
            [0, 0, 0],
        ]
        collision_scales = [0.5, 0.25, 0.5, 0.5, 0.4, 0.4, 1.0]

        # Having colorful cubes
        colors = [
        [1, 0, 0, 1],   # red
        [0, 1, 0, 1],   # green
        [0, 0, 1, 1],   # blue
        # [1, 1, 0, 1],   # yellow
        [1, 0, 1, 1],   # magenta
        # [0, 0, 0, 1],   # black
        [0, 1, 1, 1],   # cyan
        [0.8, 0.4, 0, 1], # orange
        [0, 0.5, 0.5, 1],   # teal
        ]

    for i in range(len(collision_positions)):
        uid = p.loadURDF(
            "cube.urdf",
            basePosition=collision_positions[i],
            baseOrientation=p.getQuaternionFromEuler(collision_orientations[i]),
            globalScaling=collision_scales[i]
        )
        p.changeVisualShape(uid, -1, rgbaColor=base_colors[i % len(base_colors)])
        collision_ids.append(uid)

    # Define 6-DOF Goal Positions
    # Previous single-arm goals were length 3.
    # We now need length 6: [Left_J1, L_J2, L_J3, Right_J1, R_J2, R_J3]
    # We will create a "mirror" motion or synchronized motion.
    
    # Single arm goals from original code
    single_goals = [
        [-2.54, 0.15, -0.15],
        [-1.79, 0.15, -0.15],
        [0.5, 0.15, -0.15],
        [1.7, 0.2, -0.15],
        [-2.54, 0.15, -0.15],
    ]
    
    goal_positions = []
    for g in single_goals:
        # Construct a 6D goal. 
        # Left arm does 'g'. Right arm does 'g' (but mirrored logic might be needed depending on coord system).
        # For simplicity, let's just apply 'g' to both.
        # Since the right arm is mounted differently, 'g' might look different, but it's valid joint space.
        full_goal = g + g 
        goal_positions.append(full_goal)

    print("Checking validity of goal positions...")
    valid_goals = []
    
    for i, goal in enumerate(goal_positions):
        valid = is_state_valid(
            robot_id=arm_id,
            joint_indices=movable_joints,
            q_target=goal,
            obstacle_ids=collision_ids, # Defined in Step 3
            check_self=True
        )
        
        if valid:
            valid_goals.append(goal)
            print(f"Goal {i}: VALID")
        else:
            print(f"Goal {i}: INVALID (Collision detected). Skipping.")

    # Update the list to only use valid goals
    if len(valid_goals) < 2:
        print("Error: Not enough valid goals to form a path!")
        exit()
        
    goal_positions = valid_goals
    
    # Reset robot to the first valid goal to start
    for i, joint_idx in enumerate(movable_joints):
        p.resetJointState(arm_id, joint_idx, goal_positions[0][i])

    # Joint Limits: 6 pairs of [-pi, pi] etc.
    # We replicate the original limits twice.
    single_limits = [[-np.pi, np.pi], [0, np.pi], [-np.pi, np.pi]]
    joint_limits = single_limits + single_limits 

    # Path container - initialized with start position
    path_saved = np.array([goal_positions[0]])

    # ---------------------------------------------------------
    # 5. Initialize SDF and Planner
    # ---------------------------------------------------------
    
    # Set robot to start position
    for i, joint_idx in enumerate(movable_joints):
        p.resetJointState(arm_id, joint_idx, goal_positions[0][i])
        
    sdf_env = make_pybullet_env_sdf(collision_ids, max_distance=5.5, probe_radius=0.01)

    # Optional: Visualizing the start spheres
    visualize_spherical_robot(
        arm_id, goal_positions[0], ROBOT_SPHERES, movable_joints, color=[1, 0, 0, 0.4]
    )

    # Planning Loop
    for i in range(len(goal_positions) - 1):
        q_start = goal_positions[i]
        q_goal = goal_positions[i + 1]

        print(f"\n--- Planning segment {i+1} ---")
        print(f"Start: {np.round(q_start, 2)}")
        print(f"Goal:  {np.round(q_goal, 2)}")

        rrt_planner = RRT_CBF(
            q_start,
            q_goal,
            arm_id,
            collision_ids,
            joint_limits,
            sdf_env,
            ROBOT_SPHERES,
            joint_indices=movable_joints, # PASS ALL 6 INDICES
            max_iter=5000,
            step_size=0.3, # Reduced step size slightly for higher dim space stability
            alpha=50.0,
            d_safe=0.1,
        )
        
        path_segment = rrt_planner.plan()

        if path_segment is None:
            print(f"RRT failed to find a path.")
        else:
            print(f"RRT found path with {len(path_segment)} waypoints")
            path_saved = np.vstack((path_saved, path_segment[1:]))

    # Execution Loop
    print(f"\nExecuting path with {len(path_saved)} total waypoints...")
    
    for waypoint in path_saved:
        # Move to next waypoint
        # waypoint is length 6
        
        while True:
            # Get current joint positions (ground truth)
            # We map the movable_joints indices to the result list
            true_joint_positions = []
            for j_idx in movable_joints:
                s = p.getJointState(arm_id, j_idx)
                true_joint_positions.append(s[0])
            true_joint_positions = np.array(true_joint_positions)
            
            # Displacement in 6D
            displacement_to_waypoint = waypoint - true_joint_positions
            dist = np.linalg.norm(displacement_to_waypoint)
            
            max_speed = 0.05
            
            if dist < max_speed:
                break
            else:
                # Calculate velocity vector
                velocities = (
                    np.min((dist, max_speed))
                    * displacement_to_waypoint
                    / dist
                )                    

                # Apply velocities to the specific movable joints
                for k, v in enumerate(velocities):
                    p.setJointMotorControl2(
                        bodyIndex=arm_id,
                        jointIndex=movable_joints[k], # Map index k back to PyBullet joint index
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=v,
                    )
            
            p.stepSimulation()
            # time.sleep(1.0 / 240.0) # Uncomment for real-time speed

    print("Path execution complete.")
    # Keep window open
    while p.isConnected():
        p.stepSimulation()
        time.sleep(0.1)