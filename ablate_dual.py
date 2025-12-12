import csv
import random
import time
import numpy as np
import pybullet as p
import pybullet_data
import warnings
import os 
import sys
from contextlib import contextmanager

# Custom Imports (Ensure these files are in the same directory)
from rrt_cbf_dual import RRT_CBF  # The Dual Arm Planner
from rrt_dual import RRT
from sdf import make_pybullet_env_sdf
from envs import get_env
from utils import is_state_valid, mark_goal_configurations_dual

# Helper to suppress PyBullet Output
@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()
    def _redirect_stdout(to):
        sys.stdout.close()
        os.dup2(to.fileno(), fd)
        sys.stdout = os.fdopen(fd, "w")
    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield
        finally:
            _redirect_stdout(to=old_stdout)

warnings.filterwarnings("ignore")

# Geometry / Robot Helpers
def get_joint_index_by_name(robot_id, name):
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8")
        if joint_name == name:
            return i
    return -1

def get_link_axis_and_length(robot_id, link_index, q_ref=None):
    if q_ref is None:
        for j in range(p.getNumJoints(robot_id)):
            p.resetJointState(robot_id, j, 0.0)
    link_state = p.getLinkState(robot_id, link_index, computeForwardKinematics=True)
    pos_link = np.array(link_state[0])
    orn_link = link_state[1]
    R = np.array(p.getMatrixFromQuaternion(orn_link)).reshape(3, 3)

    child_found = False
    pos_child = pos_link
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        parent_index = info[16]
        if parent_index == link_index:
            child_state = p.getLinkState(robot_id, j, computeForwardKinematics=True)
            pos_child = np.array(child_state[0])
            child_found = True
            break
    if not child_found:
        pos_child = pos_link + np.array([0.5, 0.0, 0.0])

    dir_world = pos_child - pos_link
    length = np.linalg.norm(dir_world)
    if length < 1e-6: return np.array([1.0, 0.0, 0.0]), 0.0
    dir_local = R.T @ (dir_world / length)
    return dir_local, length

def make_link_spheres_from_fk(robot_id, link_index, radius, q_ref=None, max_spacing_factor=1.5, min_spheres=2):
    axis_local, length = get_link_axis_and_length(robot_id, link_index, q_ref=q_ref)
    if length < 1e-6:
        return [{"link_index": link_index, "local_pos": [0.0, 0.0, 0.0], "radius": radius}]
    
    max_spacing = max_spacing_factor * radius
    n_spheres = int(np.ceil(length / max_spacing)) + 1
    n_spheres = max(n_spheres, min_spheres)
    spheres = []
    for i in range(n_spheres):
        t = i / (n_spheres - 1)
        local_pos = (axis_local * (length * t)).tolist()
        spheres.append({"link_index": link_index, "local_pos": local_pos, "radius": radius})
    return spheres

def add_noise_to_obstacles(collision_ids, collision_positions, collision_orientations, noise_std):
    """Add correlated XY noise to stacked cubes."""
    if noise_std <= 0.0: return
    
    stack_groups = {}
    for idx, pos in enumerate(collision_positions):
        xy_key = (round(pos[0], 3), round(pos[1], 3))
        stack_groups.setdefault(xy_key, []).append(idx)

    for xy_key, indices in stack_groups.items():
        dx = random.gauss(0, noise_std)
        dy = random.gauss(0, noise_std)
        for idx in indices:
            orig_pos = collision_positions[idx]
            new_pos = [orig_pos[0] + dx, orig_pos[1] + dy, orig_pos[2]]
            # collision_ids[0] is ground, so +1
            body_id = collision_ids[idx + 1]
            p.resetBasePositionAndOrientation(body_id, new_pos, p.getQuaternionFromEuler(collision_orientations[idx]))


def execute_path_and_check_collision_dual(arm_id, collision_ids, movable_joints, path_saved, control_dt=1.0/240.0):
    """
    Executes the path. Checks for Environment Collision AND Self-Collision.
    """
    collided = False
    
    # Reset to start of path
    start_config = path_saved[0]
    for i, j_idx in enumerate(movable_joints):
        p.resetJointState(arm_id, j_idx, start_config[i])

    for waypoint in path_saved:
        while True:
            # 1. Get current state
            true_joint_positions = []
            for j_idx in movable_joints:
                s = p.getJointState(arm_id, j_idx)
                true_joint_positions.append(s[0])
            true_joint_positions = np.array(true_joint_positions)
            
            # 2. Vector control logic
            disp = waypoint - true_joint_positions
            dist = np.linalg.norm(disp)
            
            arrival_tolerance = 0.02
            execution_speed = 0.5
            
            if dist < arrival_tolerance:
                break # Next waypoint

            # Proportional velocity
            target_speed = min(execution_speed, dist * 2.0)
            velocities = (disp / dist) * target_speed

            for k, v in enumerate(velocities):
                p.setJointMotorControl2(
                    bodyIndex=arm_id,
                    jointIndex=movable_joints[k],
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=v,
                    force=500
                )

            p.stepSimulation()

            # 3. Collision Checks
            
            # A) Environment Collision
            for obj_id in collision_ids:
                if p.getContactPoints(bodyA=arm_id, bodyB=obj_id):
                    collided = True
                    break
            
            # B) Self Collision (Arm vs Arm)
            if not collided:
                if p.getContactPoints(bodyA=arm_id, bodyB=arm_id):
                    collided = True

            if collided:
                break
            
            # Optional: Sleep if running GUI to visualize
            # time.sleep(control_dt)

        if collided:
            break

    return collided

def run_trial(env_name, noise_std, seed, gui=False):
    random.seed(seed)
    np.random.seed(seed)

    if gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    with suppress_stdout():
        ground_id = p.loadURDF("plane.urdf")
        # Load Dual Arm
        arm_id = p.loadURDF("dual_three_link_robot.urdf", [0, 0, 0], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)

    # Identify Movable Joints (Revolute)
    movable_joints = []
    num_joints = p.getNumJoints(arm_id)
    for i in range(num_joints):
        info = p.getJointInfo(arm_id, i)
        if info[2] == p.JOINT_REVOLUTE:
            movable_joints.append(i)

    # Identify EE Tips (Fixed)
    left_ee_idx = get_joint_index_by_name(arm_id, "ee_joint_left")
    right_ee_idx = get_joint_index_by_name(arm_id, "ee_joint_right")

    # Generate Spheres
    q_ref = [0.0] * num_joints
    ROBOT_SPHERES = []
    for link_idx in movable_joints:
        ROBOT_SPHERES += make_link_spheres_from_fk(
            arm_id, link_index=link_idx, radius=0.08, q_ref=q_ref, max_spacing_factor=1.0, min_spheres=2
        )

    # Load Environment
    collision_ids = [ground_id]
    collision_positions, collision_orientations, collision_scales, colors = get_env(env_name)
    for i in range(len(collision_positions)):
        uid = p.loadURDF("cube.urdf", basePosition=collision_positions[i], baseOrientation=p.getQuaternionFromEuler(collision_orientations[i]), globalScaling=collision_scales[i])
        p.changeVisualShape(uid, -1, rgbaColor=colors[i])
        collision_ids.append(uid)

    # Define Goals (6D)
    left_arm_goals = [
        [2.347, 1.058, -1.553],
        [0.727, 1.256, -1.289],
        [-0.066, 1.421, -1.487],
        [2.347, 1.058, -1.553],
    ]
    right_arm_goals = [
        [-1.289, 1.223, -1.322],
        [-0.331, 1.223, -0.264],
        [2.347, 0.727, -0.496],
        [-1.289, 1.223, -1.322],
    ]
    
    # Combine goals
    raw_goals = [l + r for l, r in zip(left_arm_goals, right_arm_goals)]
    
    # Filter valid goals
    goal_positions = []
    for g in raw_goals:
        if is_state_valid(arm_id, movable_joints, g, collision_ids, check_self=True):
            goal_positions.append(g)

    if len(goal_positions) < 2:
        p.disconnect()
        return {
            "env": env_name, "noise_std": noise_std, "seed": seed,
            "planner_success": False, "collision": False, "success": False,
            "path_length": 0, "total_nodes": 0
        }

    # Set Initial State
    for i, j_idx in enumerate(movable_joints):
        p.resetJointState(arm_id, j_idx, goal_positions[0][i])

    # Generate SDF
    sdf_env = make_pybullet_env_sdf(collision_ids, max_distance=5.5, probe_radius=0.01)

    # Joint Limits
    single_limits = [[-np.pi, np.pi], [0, np.pi], [-np.pi, np.pi]]
    joint_limits = single_limits + single_limits 

    # PLANNING
    path_saved = np.array([goal_positions[0]])
    planner_success = True
    total_nodes = 0
    
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
            joint_indices=movable_joints,
            ee_indices=[left_ee_idx, right_ee_idx], # Pass correct EE tips
            max_iter=5000,
            step_size=0.3,
            alpha=50.0,
            d_safe=0.1+noise_std*2.0,
        )

        # rrt_planner = RRT(
        #     q_start,
        #     q_goal,
        #     arm_id,
        #     collision_ids,
        #     joint_limits,
        #     movable_joints,
        #     [left_ee_idx, right_ee_idx],
        #     max_iter=10000,
        #     step_size=0.3,
        # )
        
        path_segment = rrt_planner.plan()
        total_nodes += len(rrt_planner.node_list)

        if path_segment is None:
            planner_success = False
            break
        else:
            path_saved = np.vstack((path_saved, path_segment[1:]))

    # NOISE & EXECUTION
    collided = False
    if planner_success:
        # Add Noise
        add_noise_to_obstacles(collision_ids, collision_positions, collision_orientations, noise_std)
        
        # Execute
        collided = execute_path_and_check_collision_dual(
            arm_id, collision_ids, movable_joints, path_saved
        )

    # Metrics
    if planner_success:
        diffs = np.diff(path_saved, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        path_length = float(np.sum(seg_lengths))
        num_waypoints = int(path_saved.shape[0])
    else:
        path_length = float("nan")
        num_waypoints = 0

    success = planner_success and not collided

    result = {
        "env": env_name,
        "noise_std": noise_std,
        "seed": seed,
        "planner_success": bool(planner_success),
        "collision": bool(collided),
        "success": bool(success),
        "path_length": path_length,
        "num_waypoints": num_waypoints,
        "total_nodes": int(total_nodes),
    }

    p.disconnect()
    return result

def main():
    env_names = ["simple"]
    noise_stds = [0.0, 0.005, 0.01, 0.02]
    seeds = [0, 1, 3, 6, 7]

    results = []
    num_of_trials = len(env_names) * len(noise_stds) * len(seeds)
    print(f"Running {num_of_trials} Dual Arm trials...")
    
    count = 0
    for env_name in env_names:
        for noise_std in noise_stds:
            for seed in seeds:
                count += 1
                print(f"Trial {count}/{num_of_trials}: Env={env_name}, Noise={noise_std}, Seed={seed}...", end=" ", flush=True)
                res = run_trial(env_name, noise_std, seed, gui=False)
                print(f"Success: {res['success']}")
                results.append(res)

    if not results:
        print("No results collected.")
        return

    fieldnames = list(results[0].keys())
    with open("ablation_results_cbf.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("Saved results to ablation_results_dual_cbf.csv")

if __name__ == "__main__":
    main()