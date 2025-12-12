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


def mark_dual_arm_goals(
    robot_id, left_indices, right_indices, left_ee, right_ee, goal_config
):
    """
    Applies a configuration to both arms and marks their EE positions.
    """
    # 1. Teleport Left Arm
    for k, j_idx in enumerate(left_indices):
        p.resetJointState(robot_id, j_idx, goal_config[k])

    # 2. Teleport Right Arm (Using same angles for symmetry test, or define separate if needed)
    for k, j_idx in enumerate(right_indices):
        p.resetJointState(robot_id, j_idx, goal_config[k])

    # 3. Get EE positions (Force FK update)
    left_state = p.getLinkState(robot_id, left_ee, computeForwardKinematics=True)
    right_state = p.getLinkState(robot_id, right_ee, computeForwardKinematics=True)

    pos_left = left_state[4]
    pos_right = right_state[4]

    # 4. Create Markers
    # Cyan for Left
    v_left = p.createVisualShape(p.GEOM_SPHERE, radius=0.06, rgbaColor=[0, 1, 1, 0.6])
    m_left = p.createMultiBody(baseVisualShapeIndex=v_left, basePosition=pos_left)

    # Magenta for Right
    v_right = p.createVisualShape(p.GEOM_SPHERE, radius=0.06, rgbaColor=[1, 0, 1, 0.6])
    m_right = p.createMultiBody(baseVisualShapeIndex=v_right, basePosition=pos_right)

    # 5. Add Labels
    p.addUserDebugText(
        "Left EE", [pos_left[0], pos_left[1], pos_left[2] + 0.15], [0, 1, 1]
    )
    p.addUserDebugText(
        "Right EE", [pos_right[0], pos_right[1], pos_right[2] + 0.15], [1, 0, 1]
    )

    print(f"  Left EE: {np.round(pos_left, 2)} | Right EE: {np.round(pos_right, 2)}")

    return [m_left, m_right]


if __name__ == "__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    ground_id = p.loadURDF("plane.urdf")

    robot_id = p.loadURDF(
        "dual_three_link_robot.urdf",
        [0, 0, 0],
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )

    # Identify Joint Indices Dynamically
    left_joint_names = ["baseHinge_left", "interArm_left", "interArm2_left"]
    right_joint_names = ["baseHinge_right", "interArm_right", "interArm2_right"]

    left_indices = [get_joint_index_by_name(robot_id, n) for n in left_joint_names]
    right_indices = [get_joint_index_by_name(robot_id, n) for n in right_joint_names]

    # Identify EE Indices
    ee_left_idx = get_joint_index_by_name(robot_id, "ee_joint_left")
    ee_right_idx = get_joint_index_by_name(robot_id, "ee_joint_right")

    print(f"Left Joints: {left_indices}, EE: {ee_left_idx}")
    print(f"Right Joints: {right_indices}, EE: {ee_right_idx}")

    # Camera
    p.resetDebugVisualizerCamera(
        cameraDistance=2.0,
        cameraYaw=0,
        cameraPitch=-40,
        cameraTargetPosition=[0, 0, 0.5],
    )

    # Test Configurations
    test_configs = [
        [-2.54, 0.15, -0.15],
        [-1.79, 0.15, -0.15],
        [0.0, 0.0, 0.0],  # Zero pose
        [1.57, 0.5, 0.5],  # Arbitrary pose
    ]

    try:
        while True:
            for i, config in enumerate(test_configs):
                print(f"\n--- Testing Config {i}: {config} ---")

                # Mark and Move
                marker_ids = mark_dual_arm_goals(
                    robot_id,
                    left_indices,
                    right_indices,
                    ee_left_idx,
                    ee_right_idx,
                    config,
                )

                p.stepSimulation()
                time.sleep(3.0)

                # Cleanup markers
                for mid in marker_ids:
                    p.removeBody(mid)
                p.removeAllUserDebugItems()

    except KeyboardInterrupt:
        p.disconnect()
