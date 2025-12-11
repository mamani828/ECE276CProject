import pybullet as p
import numpy as np
from utils import (
    plot_path,
    check_edge_collision,
    get_ee_position,
    plot_link_coordinate_frames,
    plot_rrt_edge,
)

def check_node_collision(robot_id, object_ids, joint_indices, joint_position, distance=0.0):
    """
    Checks for collisions between a robot and a set of objects in PyBullet.
    """
    # Set joint positions using the given joint_indices
    for j_pos, j_idx in zip(joint_position, joint_indices):
        p.resetJointState(robot_id, j_idx, float(j_pos))

    # Check each link (including base with linkIndex=-1 if desired)
    num_joints = p.getNumJoints(robot_id)
    for object_id in object_ids:
        for link_index in range(-1, num_joints):
            contact_points = p.getClosestPoints(
                bodyA=robot_id,
                bodyB=object_id,
                distance=distance,
                linkIndexA=link_index,
            )
            if contact_points:
                return True

    # Check Self-Collision (New Logic)
    # We ask PyBullet for contact points where bodyA=robot AND bodyB=robot
    # Note: This requires the URDF to not have 'collision ignore' set between these specific links,
    # or requires custom logic to ignore adjacent links (like shoulder-to-upper-arm).
    self_contacts = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
    if self_contacts:
        return True

    return False

class Node:
    def __init__(self, joint_angles):
        self.joint_angles = np.array(joint_angles)
        self.parent = None
        self.cost = 0.0

class RRT_CBF:
    def __init__(
        self,
        q_start,
        q_goal,
        robot_id,
        obstacle_ids,
        q_limits,
        env_sdf,
        spheres,
        joint_indices,
        ee_indices,
        max_iter=10000,
        step_size=0.5,
        alpha=10.0,
        d_safe=0.02,
    ):
        self.q_start = Node(q_start)
        self.q_goal = Node(q_goal)
        self.robot_id = robot_id
        self.obstacle_ids = obstacle_ids
        self.q_limits = np.array(q_limits, dtype=float)
        self.env_sdf = env_sdf
        self.spheres = spheres
        self.joint_indices = joint_indices
        self.ee_indices = ee_indices
        self.dt = 0.05
        self.u_max = 1.0
        self.max_iter = max_iter
        self.step_size = float(step_size)
        self.alpha = float(alpha)
        self.d_safe = float(d_safe)

        self.node_list = [self.q_start]
        self.dim = len(q_start)
        self.q_start.cost = 0.0

        # Determine EE indices dynamically for dual arm
        # Assuming standard setup: joint_indices are [L1, L2, L3, R1, R2, R3]
        mid_point = len(joint_indices) // 2

    def _set_robot_config(self, q):
        """Reset PyBullet joint states to configuration q."""
        q = np.asarray(q, dtype=float)
        for i, j_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, j_idx, q[i])

    def _sphere_world_pos(self, link_index, local_pos):
        """World position of a sphere attached to a link at local_pos."""
        link_state = p.getLinkState(
            self.robot_id, link_index, computeForwardKinematics=True
        )
        link_pos = link_state[4]
        link_orn = link_state[5]
        world_pos, _ = p.multiplyTransforms(
            link_pos, link_orn, local_pos, [0.0, 0.0, 0.0, 1.0]
        )
        return np.array(world_pos)

    def _sphere_jacobian(self, link_index, local_pos, q):
        """3 x n linear Jacobian for a sphere point in world frame."""
        q = np.asarray(q, dtype=float)
        zeros = [0.0] * len(q)
        J_pos, J_orn = p.calculateJacobian(
            self.robot_id, link_index, local_pos, q.tolist(), zeros, zeros
        )
        # PyBullet returns Jacobian for ALL joints. We must slice it to keep only 
        # the columns corresponding to our controlled 'joint_indices'.
        # However, if 'joint_indices' matches the robot's movable joints exactly 
        # (and in order), PyBullet usually returns the correct size or we just use it directly.
        # For safety with fixed base robots where first indices are 0:
        return np.array(J_pos)

    def _sdf_grad_world(self, x, eps=1e-3):
        """Finite-difference gradient of SDF at world point x."""
        x = np.asarray(x, dtype=float)
        grad = np.zeros(3)
        for i in range(3):
            e = np.zeros(3)
            e[i] = eps
            grad[i] = (self.env_sdf(x + e) - self.env_sdf(x - e)) / (2.0 * eps)
        return grad

    def sdf_and_grad(self, q):
        """Compute CBF values h_i(q) and their gradients dh_i/dq."""
        q = np.asarray(q, dtype=float)
        self._set_robot_config(q)

        h_list = []
        dh_list = []

        for sph in self.spheres:
            link_index = sph["link_index"]
            local_pos = sph["local_pos"]

            x = self._sphere_world_pos(link_index, local_pos)
            d = self.env_sdf(x)
            h_i = d - self.d_safe

            grad_phi = self._sdf_grad_world(x)
            J_pos = self._sphere_jacobian(link_index, local_pos, q)
            
            # Ensure Jacobian dimensions match q
            # If J_pos is larger (e.g. includes fixed joints), slice it.
            if J_pos.shape[1] > len(q):
                 # This mapping depends on your specific URDF joint mapping. 
                 # For now, assuming J_pos corresponds to movable joints if passed correctly.
                 pass

            dh_i = grad_phi @ J_pos
            h_list.append(h_i)
            dh_list.append(dh_i)

        h = np.array(h_list)
        dh = np.vstack(dh_list)
        return h, dh

    def _project_onto_cbf_constraints(self, u_des, h, dh, max_iters=10):
        """QP-like projection of velocity u_des onto CBF constraints."""
        u = u_des.copy()
        m = h.shape[0]
        for _ in range(max_iters):
            changed = False
            for i in range(m):
                a = dh[i]
                b = -self.alpha * h[i]
                val = a @ u
                if val < b:
                    denom = np.dot(a, a) + 1e-10
                    u = u + (b - val) / denom * a
                    changed = True
            if not changed:
                break
        return u

    def step(self, q_from, q_to, num_substeps=5):
        q = np.asarray(q_from, dtype=float)
        q_to = np.asarray(q_to, dtype=float)

        # Making sure that distance is max step_size
        if np.linalg.norm(q_to - q) > self.step_size:
            q_to = q + self.step_size * (q_to - q) / np.linalg.norm(q_to - q)

        for _ in range(num_substeps):
            direction = q_to - q
            L = np.linalg.norm(direction)
            if L < 1e-3:
                break

            direction /= L
            u_des = self.u_max * direction

            h, dh = self.sdf_and_grad(q)
            u_safe = self._project_onto_cbf_constraints(u_des, h, dh)

            if np.linalg.norm(u_safe) < 1e-6:
                break

            q = q + u_safe * self.dt
            q = np.clip(q, self.q_limits[:, 0], self.q_limits[:, 1])

        return q

    def _visualize_edge(self, q_from, q_to):
        """
        Manually computes FK for both configurations and draws lines for ALL end effectors.
        This ensures correct joint mapping.
        """
        # Colors: Green for Left EE, Yellow for Right EE
        colors = [[0, 1, 0], [1, 1, 0]]
        
        # 1. Get positions at q_from
        self._set_robot_config(q_from)
        pos_from = []
        for ee_idx in self.ee_indices:
            state = p.getLinkState(self.robot_id, ee_idx, computeForwardKinematics=True)
            pos_from.append(state[4]) # Index 4 is Link World Position
            
        # 2. Get positions at q_to
        self._set_robot_config(q_to)
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
                lifeTime=0 # 0 means permanent
            )

    def plan(self):
        for _ in range(self.max_iter):
            if np.random.rand() < 0.10:
                q_rand = self.q_goal.joint_angles.copy()
            else:
                q_rand = np.array(
                    [np.random.uniform(lo, hi) for (lo, hi) in self.q_limits],
                    dtype=float,
                )

            if check_node_collision(self.robot_id, self.obstacle_ids, self.joint_indices, q_rand):
                continue

            nearest = self.get_nearest_node(q_rand)
            q_new = self.step(nearest.joint_angles, q_rand)

            if np.allclose(q_new, nearest.joint_angles):
                continue

            if check_node_collision(self.robot_id, self.obstacle_ids, self.joint_indices, q_new):
                continue

            new_node = Node(q_new)
            new_node.parent = nearest
            self.node_list.append(new_node)
            
            # Draw Edge for BOTH arms
            self._visualize_edge(nearest.joint_angles, new_node.joint_angles)

            # Check if close to goal
            if np.linalg.norm(new_node.joint_angles - self.q_goal.joint_angles) <= self.step_size:
                q_goal_proj = self.step(new_node.joint_angles, self.q_goal.joint_angles)

                if check_node_collision(self.robot_id, self.obstacle_ids, self.joint_indices, q_goal_proj):
                    continue

                if np.linalg.norm(q_goal_proj - self.q_goal.joint_angles) < self.step_size:
                    goal_node = Node(q_goal_proj)
                    goal_node.parent = new_node
                    self.node_list.append(goal_node)

                    # Draw Final Edge
                    self._visualize_edge(new_node.joint_angles, goal_node.joint_angles)

                    path = []
                    node = goal_node
                    while node is not None:
                        path.append(np.asarray(node.joint_angles, dtype=float))
                        node = node.parent
                    path.reverse()
                    return np.vstack(path)

        return None

    def get_nearest_node(self, q_rand):
        dists = [np.linalg.norm(node.joint_angles - q_rand) for node in self.node_list]
        nearest_index = np.argmin(dists)
        return self.node_list[nearest_index]