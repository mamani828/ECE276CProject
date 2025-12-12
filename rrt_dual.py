import numpy as np
import pybullet as p
import time

class Node:
    def __init__(self, joint_angles):
        self.joint_angles = np.array(joint_angles)
        self.parent = None

class RRT:
    def __init__(
        self,
        q_start,
        q_goal,
        robot_id,
        obstacle_ids,
        q_limits,
        joint_indices,  # [L1, L2, L3, R1, R2, R3]
        ee_indices,     # [Left_Tip, Right_Tip]
        max_iter=10000,
        step_size=0.5,
    ):
        self.q_start = Node(q_start)
        self.q_goal = Node(q_goal)
        self.robot_id = robot_id
        self.obstacle_ids = obstacle_ids
        self.q_limits = np.array(q_limits)
        self.joint_indices = joint_indices
        self.ee_indices = ee_indices
        self.max_iter = max_iter
        self.step_size = step_size
        
        self.node_list = [self.q_start]

    def step(self, from_config, to_config):
        """Steer from 'from_config' towards 'to_config' by step_size."""
        q_from = np.array(from_config)
        q_to = np.array(to_config)
        
        direction = q_to - q_from
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return q_to
        else:
            return q_from + (direction / distance) * self.step_size

    def get_nearest_node(self, random_point):
        dists = [np.linalg.norm(node.joint_angles - random_point) for node in self.node_list]
        return self.node_list[np.argmin(dists)]

    def _check_collision(self, config):
        """
        Explicitly checks collision for a specific configuration.
        """
        # 1. Set Robot Configuration
        for i, j_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, j_idx, config[i])
        p.performCollisionDetection()

        # 2. Check Environment Collision
        for obj_id in self.obstacle_ids:
            if p.getContactPoints(bodyA=self.robot_id, bodyB=obj_id):
                return True # Collision

        # 3. Check Self Collision (Arm vs Arm)
        if p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id):
             return True

        return False

    def _check_edge_collision(self, q1, q2, steps=10):
        """
        Discretize edge and check collision at each step.
        """
        for i in range(steps + 1):
            t = i / steps
            q_interp = q1 + (q2 - q1) * t
            if self._check_collision(q_interp):
                return True
        return False

    def _visualize_edge(self, q1, q2):
        """Draws a line for both End Effectors between q1 and q2."""
        # 1. Get Start Positions
        for i, j_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, j_idx, q1[i])
        
        pos1 = []
        for ee in self.ee_indices:
            state = p.getLinkState(self.robot_id, ee, computeForwardKinematics=True)
            pos1.append(state[4]) # World pos

        # 2. Get End Positions
        for i, j_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, j_idx, q2[i])
            
        pos2 = []
        for ee in self.ee_indices:
            state = p.getLinkState(self.robot_id, ee, computeForwardKinematics=True)
            pos2.append(state[4])

        # 3. Draw Lines (Green=Left, Yellow=Right)
        colors = [[0, 1, 0], [1, 1, 0]]
        for k in range(len(self.ee_indices)):
            p.addUserDebugLine(pos1[k], pos2[k], lineColorRGB=colors[k], lineWidth=1, lifeTime=0)

    def plan(self):
        for _ in range(self.max_iter):
            # 1. Sample
            if np.random.rand() < 0.10:
                q_rand = self.q_goal.joint_angles.copy()
            else:
                q_rand = np.random.uniform(self.q_limits[:, 0], self.q_limits[:, 1])

            # 2. Nearest
            nearest_node = self.get_nearest_node(q_rand)
            
            # 3. Steer
            q_new = self.step(nearest_node.joint_angles, q_rand)

            # 4. Check Edge Collision
            if self._check_edge_collision(nearest_node.joint_angles, q_new):
                continue

            # 5. Add Node
            new_node = Node(q_new)
            new_node.parent = nearest_node
            self.node_list.append(new_node)
            
            # Visualize
            self._visualize_edge(nearest_node.joint_angles, q_new)

            # 6. Check Goal
            if np.linalg.norm(new_node.joint_angles - self.q_goal.joint_angles) <= self.step_size:
                if not self._check_edge_collision(new_node.joint_angles, self.q_goal.joint_angles):
                    
                    self.q_goal.parent = new_node
                    self._visualize_edge(new_node.joint_angles, self.q_goal.joint_angles)
                    
                    # Extract Path
                    path = []
                    curr = self.q_goal
                    while curr is not None:
                        path.append(curr.joint_angles)
                        curr = curr.parent
                    return np.array(path[::-1]) # Reverse

        return None