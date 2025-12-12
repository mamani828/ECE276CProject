# ablation_experiments.py

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
@contextmanager ##FROM GITHUB
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.


warnings.filterwarnings("ignore")
from rrt_cbf import RRT_CBF
from sdf import make_pybullet_env_sdf
from envs import get_env
from utils import mark_goal_configurations
from main import make_link_spheres_from_fk

def add_noise_to_obstacles(collision_ids,
                           collision_positions,
                           collision_orientations,
                           noise_std,
                           group_by_xy=True):
    """
    Add correlated XY noise to stacked cubes so stacks move together.
    """
    if noise_std <= 0.0:
        return collision_positions  # no change

    if group_by_xy:
        stack_groups = {}
        for idx, pos in enumerate(collision_positions):
            xy_key = (round(pos[0], 3), round(pos[1], 3))
            stack_groups.setdefault(xy_key, []).append(idx)
    else:
        # each cube independent
        stack_groups = {idx: [idx] for idx in range(len(collision_positions))}

    new_positions = [list(p_) for p_ in collision_positions]

    for xy_key, indices in stack_groups.items():
        dx = random.gauss(0, noise_std)
        dy = random.gauss(0, noise_std)

        for idx in indices:
            orig_pos = collision_positions[idx]
            new_pos = [
                orig_pos[0] + dx,
                orig_pos[1] + dy,
                orig_pos[2],  # preserve Z
            ]
            new_positions[idx] = new_pos

            # collision_ids[0] is ground, so +1 offset
            body_id = collision_ids[idx + 1]
            p.resetBasePositionAndOrientation(
                body_id,
                new_pos,
                p.getQuaternionFromEuler(collision_orientations[idx]),
            )

    return new_positions


def execute_path_and_check_collision(arm_id,
                                     collision_ids,
                                     path_saved,
                                     max_speed=0.05,
                                     control_dt=1.0 / 240.0):
    """
    Execute the planned joint path in simulation and check for any collision.
    Returns: collided (bool)
    """
    collided = False

    for waypoint in path_saved:
        # Move to next waypoint
        while True:
            # ground-truth joints
            true_joint_positions = np.array([p.getJointState(arm_id, j)[0] for j in range(3)] )
            displacement_to_waypoint = waypoint - true_joint_positions
            if np.linalg.norm(displacement_to_waypoint) < 0.01:
                # stop the robot
                p.setJointMotorControlArray(
                    bodyIndex=arm_id,
                    jointIndices=[0, 1, 2],
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocities=[0.0] * 3,
                )
                break
            else:
                # calculate the "velocity" to reach the next waypoint
                velocities = displacement_to_waypoint * 0.5               

                for joint_index, velocity in enumerate(velocities):
                    p.setJointMotorControl2(
                        bodyIndex=arm_id,
                        jointIndex=joint_index,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=velocity,)
            p.stepSimulation()

            # collision against ALL collision_ids (cubes + ground)
            for obj_id in collision_ids:
                contacts = p.getContactPoints(bodyA=arm_id, bodyB=obj_id)
                if contacts:
                    collided = True
                    break

            if collided:
                break

            time.sleep(control_dt)

        if collided:
            break

    return collided


def run_trial(env_name, noise_std, seed, gui=True):
    """
    Run a single experiment:
      - env_name: "simple" or "complex"
      - noise_std: std dev of XY noise for obstacle stacks
      - seed: random seed
      - gui: if True, use p.GUI; otherwise p.DIRECT
    Returns a dict of metrics.
    """
    random.seed(seed)
    np.random.seed(seed)

    if gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Robot and ground
    with suppress_stdout():
        ground_id = p.loadURDF("plane.urdf")
    with suppress_stdout():
        arm_id = p.loadURDF("robot.urdf", [0, 0, 0], useFixedBase=True)

    q_ref = [0.0] * p.getNumJoints(arm_id)

    # Build robot spheres once per trial
    ROBOT_SPHERES = []
    for link_idx in [0, 1, 2]:
        ROBOT_SPHERES += make_link_spheres_from_fk(
            arm_id,
            link_index=link_idx,
            radius=0.1,
            q_ref=q_ref,
            max_spacing_factor=1.0,
            min_spheres=2,
        )

    # Environment
    collision_ids = [ground_id]
    collision_positions, collision_orientations, collision_scales, colors = get_env(env_name)

    for i in range(len(collision_scales)):
        uid = p.loadURDF(
            "cube.urdf",
            basePosition=collision_positions[i],
            baseOrientation=p.getQuaternionFromEuler(collision_orientations[i]),
            globalScaling=collision_scales[i],
        )
        p.changeVisualShape(uid, -1, rgbaColor=colors[i])
        collision_ids.append(uid)

    # Goals and joint limits (same as original script)
    goal_positions = [
        [0, 0, 0],
        [-1.289, 0.562, 1.487],
        [-2.776, 0.099, 2.545],
        [-1.520, 1.785, 2.545],
    ]
    joint_limits = [[-np.pi, np.pi], [0, np.pi], [-np.pi, np.pi]]

    # Optional: only useful when using GUI
    mark_goal_configurations(arm_id, [0, 1, 2], goal_positions, 3)

    # Initial joints
    for j, qj in enumerate(goal_positions[0]):
        p.resetJointState(arm_id, j, qj)

    # SDF
    sdf_env = make_pybullet_env_sdf(
        collision_ids,
        max_distance=5.5,
        probe_radius=0.01,
    )

    # Plan through all goals
    path_saved = np.array([goal_positions[0]])
    planner_success = True
    total_nodes = 0
    from rrt import RRT
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
            max_iter=10000,
            step_size=0.5,
            alpha=50.0,
            d_safe=0.1,
        )
        # rrt_planner = RRT(q_start, q_goal, arm_id, collision_ids, joint_limits,
        #                     max_iter=5000,
        #                     step_size=0.5)
        path_segment = rrt_planner.plan()
        total_nodes += len(rrt_planner.node_list)

        if path_segment is None:
            planner_success = False
            break
        else:
            path_saved = np.vstack((path_saved, path_segment[1:]))

    # Add noise after planning (as you did originally)
    if planner_success:
        add_noise_to_obstacles(
            collision_ids,
            collision_positions,
            collision_orientations,
            noise_std=noise_std,
            group_by_xy=True,
        )

    # Execute path and check collisions
    collided = False
    if planner_success:
        # reset to first goal before execution
        for j, qj in enumerate(goal_positions[0]):
            p.resetJointState(arm_id, j, qj)
        collided = execute_path_and_check_collision(
            arm_id,
            collision_ids,
            path_saved,
        )

    # Simple metrics
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
    env_names = ["simple", "complex"]
    noise_stds = [0.0, 0.005, 0.01, 0.02]
    seeds = [0, 1, 2, 3, 4]

    results = []
    num_of_trials = len(env_names) * len(noise_stds) * len(seeds)
    print(f"Running {num_of_trials} trials...")
    for env_name in env_names:
        for noise_std in noise_stds:
            for seed in seeds:
                print(f"Trial {len(results)+1}/{num_of_trials}: ", end="")
                print(f"Running env={env_name}, noise_std={noise_std}, seed={seed}")
                res = run_trial(env_name, noise_std, seed, gui=True)
                results.append(res)

    if not results:
        print("No results collected.")
        return

    fieldnames = list(results[0].keys())
    with open("ablation_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("Saved results to ablation_results.csv")


if __name__ == "__main__":
    main()
