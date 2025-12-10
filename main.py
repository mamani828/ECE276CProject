import random
import time

import pybullet as p
import pybullet_data

from rrt import RRT
from rrt_cbf import RRT_CBF
from useful_code import *
from matplotlib import pyplot as plt
from sdf import make_pybullet_env_sdf, visualize_sdf_slice
from envs import get_env
from utils import mark_goal_configurations

def create_visual_spheres(spheres, color=[1, 0, 0, 0.4]):
    """
    Creates visual-only sphere bodies in PyBullet.
    Returns a list of body IDs corresponding to the spheres.
    """
    sphere_body_ids = []
    for sph in spheres:
        r = sph["radius"]
        
        # Create the visual shape
        v_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=r,
            rgbaColor=color
        )
        
        # Create the body (no mass, no collision, just visual)
        # We start them at [0,0,0]; they will be moved immediately
        b_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=v_id,
            basePosition=[0, 0, 0] 
        )
        sphere_body_ids.append(b_id)
        
    return sphere_body_ids

def update_visual_spheres(robot_id, sphere_body_ids, spheres):
    """
    Updates the positions of the existing visual spheres based on the 
    CURRENT robot state (simulation state).
    """
    # Cache link states to avoid calling getLinkState multiple times for the same link
    link_states = {}
    
    for i, sph in enumerate(spheres):
        link_idx = sph["link_index"]
        
        # Fetch link state only if we haven't already for this frame
        if link_idx not in link_states:
            state = p.getLinkState(robot_id, link_idx, computeForwardKinematics=True)
            link_pos = np.array(state[0])
            link_orn = state[1]
            link_rot = np.array(p.getMatrixFromQuaternion(link_orn)).reshape(3, 3)
            link_states[link_idx] = (link_pos, link_rot)
            
        # Retrieve cached state
        pos, rot = link_states[link_idx]
        local_pos = np.array(sph["local_pos"])
        
        # Calculate new world position: p_world = p_link + R_link * p_local
        world_pos = pos + rot @ local_pos
        
        # Teleport the sphere body to the new location
        p.resetBasePositionAndOrientation(sphere_body_ids[i], world_pos, [0, 0, 0, 1])

def get_link_axis_and_length(robot_id, link_index, q_ref=None):
    """Compute local axis and length of link using a reference configuration."""
    if q_ref is None:
        q_ref = [0.0] * p.getNumJoints(robot_id)
    # Set reference joint angles
    for j, qj in enumerate(q_ref):
        p.resetJointState(robot_id, j, qj)

    # Get link pose (current link)
    link_state = p.getLinkState(robot_id, link_index, computeForwardKinematics=True)
    pos_link = np.array(link_state[0])
    orn_link = link_state[1]
    R = np.array(p.getMatrixFromQuaternion(orn_link)).reshape(3, 3)

    # Approximate "child joint" as the next link’s frame
    # (works for a simple chain; adjust if your topology is different)
    if link_index + 1 < p.getNumJoints(robot_id):
        child_state = p.getLinkState(
            robot_id, link_index + 1, computeForwardKinematics=True
        )
        pos_child = np.array(child_state[0])
    else:
        # Last link: just assume a small offset along some direction
        pos_child = pos_link + np.array([0.1, 0.0, 0.0])

    # Direction and length in WORLD frame
    dir_world = pos_child - pos_link
    length = np.linalg.norm(dir_world)
    if length < 1e-6:
        # Degenerate case
        return np.array([0.0, 0.0, 1.0]), 0.0

    # Express this direction in the LINK frame: dir_local = R^T * dir_world
    dir_local = R.T @ (dir_world / length)
    return dir_local, length


def make_link_spheres_from_fk(
    robot_id, link_index, radius, q_ref=None, max_spacing_factor=1.5, min_spheres=2
):
    axis_local, length = get_link_axis_and_length(robot_id, link_index, q_ref=q_ref)

    # If the link is degenerate, just place one sphere at the link frame
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


if __name__ == "__main__":
    # Problem setup
    env = "complex"

    # Initialize PyBullet
    # p.connect(p.GUI)
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # For default URDFs
    p.setGravity(0, 0, -9.8)

    # Load the plane and robot arm
    ground_id = p.loadURDF("plane.urdf")
    arm_id = p.loadURDF("robot.urdf", [0, 0, 0], useFixedBase=True)
    # After loading arm_id
    q_ref = [0.0] * p.getNumJoints(arm_id)

    ROBOT_SPHERES = []
    for link_idx in [0, 1, 2]:
        ROBOT_SPHERES += make_link_spheres_from_fk(
            arm_id,
            link_index=link_idx,
            radius=0.1,
            q_ref=q_ref,
            max_spacing_factor=1.0,  # tune overlap
            min_spheres=2,  # at least 2 spheres per link
        )

    # Add Collision Objects
    collision_ids = [ground_id]  # add the ground to the collision list
    collision_positions, collision_orientations, collision_scales, colors = get_env(env)

    for i in range(len(collision_scales)):
        uid = p.loadURDF(
            "cube.urdf",
            basePosition=collision_positions[i],
            baseOrientation=p.getQuaternionFromEuler(collision_orientations[i]),
            globalScaling=collision_scales[i]
        )
        
        p.changeVisualShape(uid, -1, rgbaColor=colors[i])
        collision_ids.append(uid)

    # Goal Joint Positions for the Robot
    goal_positions = [
        [-2.54, 0.15, -0.15],
        [-1.79, 0.15, -0.15],
        [0.5, 0.15, -0.15],
        [1.7, 0.2, -0.15],
    ]

    # Joint Limits of the Robot
    joint_limits = [[-np.pi, np.pi], [0, np.pi], [-np.pi, np.pi]]

    # A3xN path array that will be filled with waypoints through all the goal positions
    path_saved = np.array([[-2.54, 0.15, -0.15]])  # Start at the first goal position

    # Run the simulation and move the robot along the saved path
    # Mark the goal configurations
    mark_goal_configurations(arm_id, [0, 1, 2], goal_positions, 2)
    # Set the initial joint positions
    for joint_index, joint_pos in enumerate(goal_positions[0]):
        p.resetJointState(arm_id, joint_index, joint_pos)
    sdf_env = make_pybullet_env_sdf(collision_ids, max_distance=5.5, probe_radius=0.01)
    
    # for height in [0.2, 0.4, 0.6, 0.8, 1.0]:
    #     visualize_sdf_slice(sdf_env, height=height)

    for i in range(len(goal_positions) - 1):
        q_start = goal_positions[i]
        q_goal = goal_positions[i + 1]

        rrt_planner = RRT_CBF(
            q_start,
            q_goal,
            arm_id,
            collision_ids,
            joint_limits,
            sdf_env,
            ROBOT_SPHERES,
            joint_indices=[0, 1, 2],
            max_iter=5000,
            step_size=0.5,
            alpha=50.0,
            d_safe=0.15,
        )
        path_segment = (
            rrt_planner.plan()
        )
        
        # rrt_planner = RRT(q_start, q_goal, arm_id, collision_ids, joint_limits,
        #                     max_iter=5000,
        #                     step_size=0.5)
        # path_segment = rrt_planner.plan()

        if path_segment is None:
            print(f"RRT failed to find a path from {q_start} to {q_goal}")
        else:
            print(
                f"RRT found a path from {q_start} to {q_goal} with {len(path_segment)} waypoints"
            )
            path_saved = np.vstack((path_saved, path_segment[1:]))
    # Adding noise to the environment
    stack_groups = {}
    for idx, pos in enumerate(collision_positions):
        # Rounding prevents floating point errors from separating stacked items
        xy_key = (round(pos[0], 3), round(pos[1], 3))
        if xy_key not in stack_groups:
            stack_groups[xy_key] = []
        stack_groups[xy_key].append(idx)

    # Generate and Apply Noise
    # Adjust these standard deviations to control noise magnitude
    noise_std = 0.05
    for xy_key, indices in stack_groups.items():
        # Generate ONE noise vector for this specific stack/location
        # This ensures all cubes in a stack move together
        dx = random.gauss(0, noise_std)
        dy = random.gauss(0, noise_std)
        d_yaw = random.gauss(0, noise_std)

        for idx in indices:
            # Retrieve original hardcoded values
            orig_pos = collision_positions[idx]
            orig_euler = collision_orientations[idx]

            # Calculate new position (Preserve Z)
            new_pos = [
                orig_pos[0] + dx, 
                orig_pos[1] + dy, 
                orig_pos[2]
            ]

            # Calculate new orientation (Add noise only to Yaw/Z-rotation)
            new_euler = [
                orig_euler[0], 
                orig_euler[1], 
                orig_euler[2] + d_yaw
            ]
            new_quat = p.getQuaternionFromEuler(new_euler)

            # Map index to pybullet body ID (collision_ids[0] is ground, so +1)
            body_id = collision_ids[idx + 1]

            # Apply the transform
            p.resetBasePositionAndOrientation(body_id, new_pos, new_quat)

    # Move through the waypoints
    print(f"Number of nodes {len(rrt_planner.node_list)}")
    live_sphere_ids = create_visual_spheres(ROBOT_SPHERES, color=[0, 1, 0, 0.3])
    
    for joint_index, joint_pos in enumerate(goal_positions[0]):
        p.resetJointState(arm_id, joint_index, joint_pos)
    for waypoint in path_saved:
        # "move" to next waypoints
        for joint_index, joint_pos in enumerate(waypoint):
            # run velocity control until waypoint is reached
            while True:
                # get current joint positions (ground truth)
                true_joint_positions = np.array([p.getJointState(arm_id, i)[0] for i in range(3)])
                
                # calculate the displacement to reach the next waypoint using MEASURED positions
                displacement_to_waypoint = waypoint - true_joint_positions
                
                # check if goal is reached
                max_speed = 0.05
                if np.linalg.norm(displacement_to_waypoint) < max_speed:
                    break
                else:
                    # calculate the "velocity" to reach the next waypoint
                    velocities = (
                        np.min((np.linalg.norm(displacement_to_waypoint), max_speed))
                        * displacement_to_waypoint
                        / np.linalg.norm(displacement_to_waypoint)
                    )                    

                    for joint_index, velocity in enumerate(velocities):
                        p.setJointMotorControl2(
                            bodyIndex=arm_id,
                            jointIndex=joint_index,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=velocity,
                        )
                # Visualize the robot at the current waypoint
                # visualize_spherical_robot(
                #     arm_id, waypoint, ROBOT_SPHERES, color=[1, 0, 0, 0.4]
                # )
                # Take a simulation step
                p.stepSimulation()
                # We check contact against ALL collision_ids (cubes + ground)
                for obj_id in collision_ids:
                    # getContactPoints returns a list if contact exists, None otherwise
                    contacts = p.getContactPoints(bodyA=arm_id, bodyB=obj_id)
                    if contacts:
                        print(f"Collision detected with obstacle ID {obj_id}")
                        exit()
                # Update the visual spheres to the current robot state
                update_visual_spheres(arm_id, live_sphere_ids, ROBOT_SPHERES)
        time.sleep(1.0 / 240.0)

    # Disconnect from PyBullet
    p.disconnect()
