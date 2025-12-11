import pybullet as p
import pybullet_data
import time
import numpy as np

# --- CONFIGURATION ---
ENV_TYPE = "complex"  # Change to "simple" or "complex"
# ---------------------

def get_joint_index_by_name(robot_id, name):
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8")
        if joint_name == name:
            return i
    return -1

def get_env(env_name: str):
    """
    Your custom environment definition.
    """
    positions = []
    orientations = []
    scales = []
    colors = []

    if env_name == "simple":
        positions = [
            [0.3, 0.5, 0.251],
            [-0.3, 0.3, 0.101],
            [-1, -0.15, 0.251],
            [-1, -0.15, 0.752],
            [-0.5, -1, 0.251],
            [0.5, -0.35, 0.201],
            [0.5, -0.35, 0.602],
        ]
        orientations = [
            [0, 0, 0.5],
            [0, 0, 0.2],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0.25],
            [0, 0, 0.5],
        ]
        scales = [0.5, 0.25, 0.5, 0.5, 0.5, 0.4, 0.4]
        colors = [
            [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1],
            [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1], [0.8, 0.4, 0, 1],
        ]
    
    if env_name == "complex":
        positions = [
            [0.3, 0.5, 0.251], [-0.3, 0.3, 0.101], [-1, -0.15, 0.251],
            [-0.5, -1, 0.251], [0.5, -0.35, 0.201], [0.5, -0.35, 0.602],
            [-0.75, -0.575, 1.003],
        ]
        orientations = [
            [0, 0, 0.5], [0, 0, 0.2], [0, 0, 0],
            [0, 0, 0], [0, 0, 0.25], [0, 0, 0.5], [0, 0, 0],
        ]
        scales = [0.5, 0.25, 0.5, 0.5, 0.4, 0.4, 1.0]
        colors = [
            [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1],
            [1, 0, 1, 1], [0, 1, 1, 1], [0.8, 0.4, 0, 1], [0, 0.5, 0.5, 1],
        ]
    
    return positions, orientations, scales, colors

def load_environment(env_name):
    print(f"Loading '{env_name}' environment...")
    positions, orientations, scales, colors = get_env(env_name)
    
    obstacle_ids = []
    for i in range(len(positions)):
        uid = p.loadURDF(
            "cube.urdf",
            basePosition=positions[i],
            baseOrientation=p.getQuaternionFromEuler(orientations[i]),
            globalScaling=scales[i]
        )
        p.changeVisualShape(uid, -1, rgbaColor=colors[i])
        obstacle_ids.append(uid)
    return obstacle_ids

def main():
    # 1. Setup PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # 2. Load Assets
    ground_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("dual_three_link_robot.urdf", [0, 0, 0], useFixedBase=True)
    
    # 3. Load Selected Environment
    load_environment(ENV_TYPE)

    # 4. Identify Joints & EEs
    # Left Chain
    left_joints = ["baseHinge_left", "interArm_left", "interArm2_left"]
    left_indices = [get_joint_index_by_name(robot_id, n) for n in left_joints]
    
    # Right Chain
    right_joints = ["baseHinge_right", "interArm_right", "interArm2_right"]
    right_indices = [get_joint_index_by_name(robot_id, n) for n in right_joints]

    # End Effectors
    left_ee_idx = get_joint_index_by_name(robot_id, "ee_joint_left")
    right_ee_idx = get_joint_index_by_name(robot_id, "ee_joint_right")

    if left_ee_idx == -1 or right_ee_idx == -1:
        print("Warning: EE joints not found!")

    # Combine for control loop
    all_movable = left_indices + right_indices
    all_names = left_joints + right_joints

    # 5. Create GUI Sliders
    sliders = []
    for i, name in enumerate(all_names):
        # Range -Pi to +Pi
        sid = p.addUserDebugParameter(f"{name}", -3.14, 3.14, 0.0)
        sliders.append(sid)

    # Camera View
    p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=0, cameraPitch=-50, cameraTargetPosition=[0, 0, 0.5])

    # 6. Visual Markers (Ghost EEs)
    # Left (Cyan)
    v_l = p.createVisualShape(p.GEOM_SPHERE, radius=0.06, rgbaColor=[0, 1, 1, 0.8])
    marker_l = p.createMultiBody(baseVisualShapeIndex=v_l, basePosition=[0,0,0])
    
    # Right (Magenta)
    v_r = p.createVisualShape(p.GEOM_SPHERE, radius=0.06, rgbaColor=[1, 0, 1, 0.8])
    marker_r = p.createMultiBody(baseVisualShapeIndex=v_r, basePosition=[0,0,0])

    print("\n---------------------------------------------------------")
    print(f"Dual Arm Goal Tuner: {ENV_TYPE}")
    print("Adjust sliders. Left config first, then Right config.")
    print("---------------------------------------------------------\n")

    last_print_time = time.time()

    try:
        while True:
            # Read Sliders
            current_config = []
            for sid in sliders:
                val = p.readUserDebugParameter(sid)
                current_config.append(val)

            # Update Robot
            for i, j_idx in enumerate(all_movable):
                p.resetJointState(robot_id, j_idx, current_config[i])

            # Calculate EE Positions (FK)
            l_state = p.getLinkState(robot_id, left_ee_idx, computeForwardKinematics=True)
            r_state = p.getLinkState(robot_id, right_ee_idx, computeForwardKinematics=True)
            
            l_pos = l_state[4]
            r_pos = r_state[4]

            # Update Ghost Markers
            p.resetBasePositionAndOrientation(marker_l, l_pos, [0,0,0,1])
            p.resetBasePositionAndOrientation(marker_r, r_pos, [0,0,0,1])

            # Collision Check
            p.performCollisionDetection()
            in_collision = False
            contacts = p.getContactPoints(bodyA=robot_id)
            for c in contacts:
                if c[2] != ground_id:
                    in_collision = True
                    break
            
            status_text = "COLLISION!" if in_collision else "SAFE"
            text_color = [1, 0, 0] if in_collision else [0, 1, 0]

            # Debug Text
            p.removeAllUserDebugItems()
            p.addUserDebugText(f"L", [l_pos[0], l_pos[1], l_pos[2]+0.2], [0,1,1], textSize=1.5)
            p.addUserDebugText(f"R", [r_pos[0], r_pos[1], r_pos[2]+0.2], [1,0,1], textSize=1.5)
            p.addUserDebugText(status_text, [0, 0, 1.0], text_color, textSize=2.0)

            # Print Config for Copy-Paste
            if time.time() - last_print_time > 0.5:
                # Format: [L1, L2, L3, R1, R2, R3]
                config_str = "[" + ", ".join([f"{x:.3f}" for x in current_config]) + "]"
                
                print(f"Config: {config_str} | Status: {status_text}")
                last_print_time = time.time()

            p.stepSimulation()
            time.sleep(1.0/60.0)

    except KeyboardInterrupt:
        p.disconnect()

if __name__ == "__main__":
    main()