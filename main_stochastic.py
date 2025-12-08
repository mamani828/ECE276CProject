import random
import time
import numpy as np
import pybullet as p
import pybullet_data
from rrt_cbf_stochastic import RRT_CBF_Stochastic
from sdf import make_pybullet_env_sdf, visualize_sdf_slice

# Helper functions
def get_sphere_centers(robot_id, joint_q, spheres):
    for j, qj in enumerate(joint_q):
        p.resetJointState(robot_id, j, float(qj))
    centers = []
    for sph in spheres:
        link_idx = sph["link_index"]
        local_pos = np.asarray(sph["local_pos"], dtype=float)
        link_state = p.getLinkState(robot_id, link_idx, computeForwardKinematics=True)
        link_pos = np.asarray(link_state[0], dtype=float)
        link_orn = link_state[1]
        R = np.array(p.getMatrixFromQuaternion(link_orn)).reshape(3, 3)
        world_pos = link_pos + R @ local_pos
        centers.append(world_pos)
    return centers

def visualize_spherical_robot(robot_id, joint_q, spheres, color=[1, 0, 0, 0.4]):
    """
    Draw translucent spheres for the spherical approximation of the robot.
    """
    centers = get_sphere_centers(robot_id, joint_q, spheres)

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
    if q_ref is None: q_ref = [0.0] * p.getNumJoints(robot_id)
    for j, qj in enumerate(q_ref): p.resetJointState(robot_id, j, qj)
    link_state = p.getLinkState(robot_id, link_index, computeForwardKinematics=True)
    pos_link = np.array(link_state[0])
    R = np.array(p.getMatrixFromQuaternion(link_state[1])).reshape(3, 3)
    if link_index + 1 < p.getNumJoints(robot_id):
        child_state = p.getLinkState(robot_id, link_index + 1, computeForwardKinematics=True)
        pos_child = np.array(child_state[0])
    else:
        pos_child = pos_link + np.array([0.1, 0.0, 0.0])
    dir_world = pos_child - pos_link
    length = np.linalg.norm(dir_world)
    if length < 1e-6: return np.array([0.0, 0.0, 1.0]), 0.0
    dir_local = R.T @ (dir_world / length)
    return dir_local, length

def make_link_spheres_from_fk(robot_id, link_index, radius, q_ref=None, max_spacing_factor=1.5, min_spheres=2):
    axis_local, length = get_link_axis_and_length(robot_id, link_index, q_ref=q_ref)
    if length < 1e-6: return [{"link_index": link_index, "local_pos": [0,0,0], "radius": radius}]
    max_spacing = max_spacing_factor * radius
    n_spheres = max(int(np.ceil(length / max_spacing)) + 1, min_spheres)
    spheres = []
    for i in range(n_spheres):
        t = i / (n_spheres - 1)
        local_pos = (axis_local * (length * t)).tolist()
        spheres.append({"link_index": link_index, "local_pos": local_pos, "radius": radius})
    return spheres

# Extended Kalman Filter
class SimpleEKF:
    def __init__(self, x_init, P_init, Q, R):
        self.x = np.array(x_init) # State estimate
        self.P = np.array(P_init) # Covariance estimate
        self.Q = Q # Process Noise Covariance
        self.R = R # Measurement Noise Covariance

    def predict(self, u, dt):
        """
        x_k+1 = x_k + u*dt (Simple kinematic model)
        P_k+1 = F P_k F^T + Q (F is identity for linear kinematics)
        """
        self.x = self.x + u * dt
        self.P = self.P + self.Q 

    def update(self, z):
        """
        Correction step using measurement z
        K = P H^T (H P H^T + R)^-1  (H is identity if measuring states directly)
        x = x + K(z - x)
        P = (I - KH) P
        """
        # H is Identity (we measure joints directly)
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        y = z - self.x # Residual
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K) @ self.P

# Main Loop
if __name__ == "__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    ground_id = p.loadURDF("plane.urdf")
    arm_id = p.loadURDF("robot.urdf", [0, 0, 0], useFixedBase=True)
    
    # Setup Robot Approximation
    q_ref = [0.0] * p.getNumJoints(arm_id)
    ROBOT_SPHERES = []
    for link_idx in [0, 1, 2]:
        ROBOT_SPHERES += make_link_spheres_from_fk(arm_id, link_idx, 0.08, q_ref, 1.0, 2)

    # Setup Obstacles
    collision_ids = [ground_id]
    collision_positions = [[0.3, 0.5, 0.251], [-0.3, 0.3, 0.101], [-1, -0.15, 0.251], 
                           [-1, -0.15, 0.752], [-0.5, -1, 0.251], [0.5, -0.35, 0.201], [0.5, -0.35, 0.602]]
    collision_orientations = [[0, 0, 0.5], [0, 0, 0.2], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0.25], [0, 0, 0.5]]
    collision_scales = [0.5, 0.25, 0.5, 0.5, 0.5, 0.4, 0.4]
    
    colors = [[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1], [1,0,1,1], [0,1,1,1], [0.8,0.4,0,1]]
    for i in range(len(collision_scales)):
        uid = p.loadURDF("cube.urdf", basePosition=collision_positions[i],
                         baseOrientation=p.getQuaternionFromEuler(collision_orientations[i]),
                         globalScaling=collision_scales[i])
        p.changeVisualShape(uid, -1, rgbaColor=colors[i])
        collision_ids.append(uid)

    goal_positions = [[-2.54, 0.15, -0.15], [-1.79, 0.15, -0.15], [0.5, 0.15, -0.15], 
                      [1.7, 0.2, -0.15], [-2.54, 0.15, -0.15]]
    joint_limits = [[-np.pi, np.pi], [0, np.pi], [-np.pi, np.pi]]

    # Initialize Robot
    for j_idx, j_pos in enumerate(goal_positions[0]):
        p.resetJointState(arm_id, j_idx, j_pos)
    
    # Create SDF Environment
    sdf_env = make_pybullet_env_sdf(collision_ids, max_distance=5.5, probe_radius=0.01)
    
    # Planning Phase (Robust RRT)
    path_saved = np.array([goal_positions[0]])
    
    # Noise Configuration for Planning (Assumptions)
    PLAN_PROCESS_NOISE = 0.02
    
    for i in range(len(goal_positions) - 1):
        q_start = goal_positions[i]
        q_goal = goal_positions[i + 1]

        # Use the Robust RRT (Stochastic CBF)
        rrt_planner = RRT_CBF_Stochastic(
            q_start, q_goal, arm_id, collision_ids, joint_limits, sdf_env, ROBOT_SPHERES,
            joint_indices=[0, 1, 2], max_iter=2000, step_size=0.4, alpha=30.0, d_safe=0.10,
            confidence_level=0.95, # High confidence required
            process_noise_std=PLAN_PROCESS_NOISE
        )
        
        path_segment = rrt_planner.plan()
        
        if path_segment is None:
            print(f"Failed segment {i}")
        else:
            print(f"Segment {i}: Found path with {len(path_segment)} nodes")
            path_saved = np.vstack((path_saved, path_segment[1:]))

    # Execution Phase (Noisy Simulation + EKF + SCBF)
    SENSOR_NOISE_STD = 0.05
    ACTUATION_NOISE_STD = 0.1
    DT_SIM = 1.0 / 240.0
    
    # Initialize EKF
    ekf = SimpleEKF(
        x_init=goal_positions[0], 
        P_init=np.eye(3)*0.01, 
        Q=np.eye(3)*(ACTUATION_NOISE_STD**2), 
        R=np.eye(3)*(SENSOR_NOISE_STD**2)
    )

    print("Starting Execution...")
    for waypoint in path_saved:
        while True:
            # Real World (Simulation)
            true_joint_positions = np.array([p.getJointState(arm_id, i)[0] for i in range(3)])
            
            # Sensing (Noisy)
            measured_joint_positions = true_joint_positions + np.random.normal(0, SENSOR_NOISE_STD, 3)
            
            # State Estimation (EKF)
            ekf.update(measured_joint_positions)
            estimated_state = ekf.x
            estimated_cov = ekf.P
            
            # Control (Based on Estimate)
            displacement = waypoint - estimated_state
            dist = np.linalg.norm(displacement)
            max_speed = 0.05
            
            if dist < max_speed:
                break
            
            # Nominal Control
            u_nominal = max_speed * displacement / dist
            
            # Active Safety Filter (Stochastic CBF)
            # We use the EKF's estimated covariance (P) to ensure safety
            h, dh = rrt_planner.sdf_and_grad(estimated_state) # Reusing helper from planner
            
            # Project u_nominal using current estimate uncertainty
            u_safe = rrt_planner._project_onto_scbf_constraints(
                u_nominal, h, dh, sigma_q=estimated_cov, max_iters=5
            )

            # Actuation (Noisy)
            noisy_velocity = u_safe + np.random.normal(0, ACTUATION_NOISE_STD, 3)
            
            # Send to robot
            for j_idx, vel in enumerate(noisy_velocity):
                p.setJointMotorControl2(arm_id, j_idx, p.VELOCITY_CONTROL, targetVelocity=vel)
            
            # EKF Predict Step
            ekf.predict(u_safe, DT_SIM) # We predict based on what we INTENDED to send (u_safe)

            p.stepSimulation()

    p.disconnect()