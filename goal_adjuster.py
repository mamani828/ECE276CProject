import pybullet as p
import pybullet_data
import time
import numpy as np

from envs import get_env
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
    robot_id = p.loadURDF("robot.urdf", [0, 0, 0], useFixedBase=True)
    
    # 3. Load Selected Environment
    load_environment(ENV_TYPE)

    # 4. Identify Joints
    # We control the 3 revolute joints
    joint_names = ["baseHinge", "interArm", "interArm2"]
    movable_indices = [get_joint_index_by_name(robot_id, name) for name in joint_names]
    
    # We track the 'ee_joint' (Tip)
    ee_idx = get_joint_index_by_name(robot_id, "ee_joint")
    if ee_idx == -1: 
        print("Warning: 'ee_joint' not found, defaulting to last link.")
        ee_idx = p.getNumJoints(robot_id) - 1

    # 5. Create GUI Sliders
    sliders = []
    for i, name in enumerate(joint_names):
        # Range -Pi to +Pi
        sid = p.addUserDebugParameter(f"Joint {i} ({name})", -3.14, 3.14, 0.0)
        sliders.append(sid)

    # Camera View
    p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=0, cameraPitch=-50, cameraTargetPosition=[0, 0, 0.5])

    # 6. Visual Marker (Ghost EE)
    v_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 1, 0.8])
    marker_id = p.createMultiBody(baseVisualShapeIndex=v_id, basePosition=[0,0,0])

    print("\n---------------------------------------------------------")
    print(f"Goal Tuner Loaded: {ENV_TYPE}")
    print("Adjust sliders to move the robot.")
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
            for i, j_idx in enumerate(movable_indices):
                p.resetJointState(robot_id, j_idx, current_config[i])

            # Calculate EE Position (FK)
            ee_state = p.getLinkState(robot_id, ee_idx, computeForwardKinematics=True)
            ee_pos = ee_state[4] # [4] is the Link Frame position

            # Update Ghost Marker
            p.resetBasePositionAndOrientation(marker_id, ee_pos, [0,0,0,1])

            # Collision Check (Optional Visual Feedback)
            p.performCollisionDetection()
            # Check contact with anything other than ground (ID 0)
            in_collision = False
            contacts = p.getContactPoints(bodyA=robot_id)
            for c in contacts:
                if c[2] != ground_id: # c[2] is bodyB
                    in_collision = True
                    break
            
            status_text = "COLLISION!" if in_collision else "SAFE"
            text_color = [1, 0, 0] if in_collision else [0, 1, 0]

            # Debug Text
            p.removeAllUserDebugItems()
            p.addUserDebugText(
                f"EE: {np.round(ee_pos, 2)}", 
                [ee_pos[0], ee_pos[1], ee_pos[2] + 0.2], 
                textSize=1.5, 
                textColorRGB=[0,0,0]
            )
            p.addUserDebugText(
                status_text, 
                [ee_pos[0], ee_pos[1], ee_pos[2] + 0.35], 
                textSize=1.5, 
                textColorRGB=text_color
            )

            # Print Config for Copy-Paste
            if time.time() - last_print_time > 0.5:
                config_str = f"[{current_config[0]:.3f}, {current_config[1]:.3f}, {current_config[2]:.3f}]"
                print(f"Config: {config_str} | EE: {np.round(ee_pos, 2)} | {status_text}")
                last_print_time = time.time()

            p.stepSimulation()
            time.sleep(1.0/60.0)

    except KeyboardInterrupt:
        p.disconnect()

if __name__ == "__main__":
    main()