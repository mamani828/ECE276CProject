import numpy as np
import pybullet as p
from utils import (
    get_ee_position,
    plot_link_coordinate_frames,
    check_edge_collision,
    plot_rrt_edge,
)

class Node:
    def __init__(self, joint_angles):
        self.joint_angles = np.array(joint_angles)  # 6D config
        self.parent = None
        self.cost = 0.0

class RRT:
    def __init__(
        self,
        q_start,
        q_goal,
        robot_id,
        obstacle_ids,
        q_limits,
        joint_indices,
        ee_indices,
        max_iter=10000,
        step_size=0.5,
    ):
        """
        RRT Initialization for Dual Arm Robot.
        """
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

        self.dim = len(q_start)
        self.q_start.cost = 0.0

    def step(self, from_node, to_joint_angles):
        """Step from "from_node" to "to_joint_angles", that should
        (a) return the to_joint_angles if it is within the self.step_size or
        (b) only step so far as self.step_size, returning the new node within that distance
        """
        q_from = np.asarray(from_node, dtype=float)
        q_to = np.asarray(to_joint_angles, dtype=float)
        if np.linalg.norm(q_to - q_from) >= self.step_size:
            q_to = q_from + (self.step_size / np.linalg.norm(q_to - q_from)) * (q_to - q_from)

        d = q_to - q_from
        L = np.linalg.norm(d)
        if L == 0 or L <= self.step_size:
            return q_to.copy()
        q_new = q_from + (self.step_size / L) * d
        return q_new

    def get_nearest_node(self, random_point):
        random_point = np.asarray(random_point, dtype=float)
        dists = [
            np.linalg.norm(node.joint_angles - random_point) for node in self.node_list
        ]
        return self.node_list[int(np.argmin(dists))]

    def _visualize_edge(self, q_from, q_to):
        """
        Helper to plot edges for BOTH end effectors.
        Manually computes FK to ensure correct visualization.
        """
        colors = [[0, 1, 0], [1, 1, 0]] # Green (Left), Yellow (Right)
        
        # 1. Get positions at q_from
        for i, j_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, j_idx, q_from[i])
            
        pos_from = []
        for ee_idx in self.ee_indices:
            state = p.getLinkState(self.robot_id, ee_idx, computeForwardKinematics=True)
            pos_from.append(state[4]) 
            
        # 2. Get positions at q_to
        for i, j_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, j_idx, q_to[i])
            
        pos_to = []
        for ee_idx in self.ee_indices:
            state = p.getLinkState(self.robot_id, ee_idx, computeForwardKinematics=True)
            pos_to.append(state[4])

        # 3. Draw Lines
        for k in range(len(self.ee_indices)):
            c = colors[k % len(colors)]
            p.addUserDebugLine(
                lineFromXYZ=pos_from[k],
                lineToXYZ=pos_to[k],
                lineColorRGB=c,
                lineWidth=1,
                lifeTime=0 
            )

    def check_collision_custom(self, q1, q2):
        """
        Wrapper to call your utility check_edge_collision.
        Ensure your utility handles self-collision if needed.
        """
        # Note: standard check_edge_collision usually takes start/end configs
        # and discretizes between them.
        return check_edge_collision(
            self.robot_id,
            self.obstacle_ids,
            q1.tolist(),
            q2.tolist(),
            discretization_step=0.05 # Finer step for dual arm safety
        )

    def plan(self):
        """Standard RRT Planning Loop"""
        for _ in range(self.max_iter):
            # 1. Sample
            if np.random.rand() < 0.10:
                q_rand = self.q_goal.joint_angles.copy()
            else:
                q_rand = np.array(
                    [np.random.uniform(lo, hi) for (lo, hi) in self.q_limits],
                    dtype=float,
                )

            # 2. Nearest
            nearest = self.get_nearest_node(q_rand)

            # 3. Steer
            q_new = self.step(nearest.joint_angles, q_rand)

            # 4. Collision Check (Edge)
            if self.check_collision_custom(nearest.joint_angles, q_new):
                continue
            
            # 5. Add to Tree
            new_node = Node(q_new)
            new_node.parent = nearest
            self.node_list.append(new_node)
            
            # Visualize
            self._visualize_edge(nearest.joint_angles, q_new)

            # 6. Check Goal
            if np.linalg.norm(new_node.joint_angles - self.q_goal.joint_angles) <= self.step_size:
                if not self.check_collision_custom(new_node.joint_angles, self.q_goal.joint_angles):
                    
                    # Add Goal Node
                    self.q_goal.parent = new_node
                    self._visualize_edge(new_node.joint_angles, self.q_goal.joint_angles)
                    
                    # Backtrack Path
                    path = []
                    node = self.q_goal
                    while node is not None:
                        path.append(np.asarray(node.joint_angles, dtype=float))
                        node = node.parent
                    path.reverse()
                    return np.vstack(path)
        
        return None