import time
import pybullet as p
import pybullet_data
import numpy as np

def get_joint_index_by_name(robot_id, name):
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8")
        if joint_name == name:
            return i
    return -1

# --- Setup ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load plane and robot
ground_id = p.loadURDF("plane.urdf")
# Ensure the fixed joint is preserved (sometimes PyBullet merges fixed links)
arm_id = p.loadURDF("robot.urdf", [0, 0, 0], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)

# Goals
goal_positions = [
    [-2.54, 0.15, -0.15],
    [-1.79, 0.15, -0.15],
]

# --- KEY FIX: Find the correct index for 'ee_joint' ---
# Based on your URDF, this should be index 3
ee_link_index = get_joint_index_by_name(arm_id, "ee_joint")
print(f"Found EE Link Index: {ee_link_index}")

if ee_link_index == -1:
    print("Error: Could not find joint named 'ee_joint'. Check URDF.")
    exit()

# Camera setup
p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0.5])

try:
    while True:
        for i, q_goal in enumerate(goal_positions):
            print(f"Goal {i}: {q_goal}")

            # 1. Teleport
            # Note: We only reset the first 3 joints (the revolute ones)
            # Joint 3 (ee_joint) is fixed, so we don't set it.
            for j in range(3): 
                p.resetJointState(arm_id, j, q_goal[j])

            # 2. Get FK for the EE Link (Index 3)
            ee_state = p.getLinkState(arm_id, ee_link_index, computeForwardKinematics=True)
            pos_ee = ee_state[4] # [4] is the world link frame position

            # 3. Spawn Marker
            v_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[0, 1, 1, 0.6])
            m_id = p.createMultiBody(baseVisualShapeIndex=v_id, basePosition=pos_ee)

            p.stepSimulation()
            time.sleep(2.0)
            
            # Cleanup
            p.removeBody(m_id)

except KeyboardInterrupt:
    p.disconnect()