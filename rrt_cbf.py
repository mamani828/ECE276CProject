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

    Args:
        robot_id (int): ID of the robot in PyBullet.
        object_ids (list[int]): IDs of obstacles in PyBullet.
        joint_indices (list[int]): PyBullet joint indices corresponding to joint_position.
        joint_position (array-like): Joint positions (same length as joint_indices).
        distance (float): Distance threshold for getClosestPoints (0.0 = actual contact).

    Returns:
        bool: True if a collision is detected, False otherwise.
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
        self.env_sdf = env_sdf  # SDF function
        self.spheres = spheres  # list of dicts: {"link_index", "local_pos"}
        self.joint_indices = joint_indices  # PyBullet joint indices for q
        self.dt = 0.05  # time step
        self.u_max = 1.0  # max joint velocity
        self.max_iter = max_iter
        self.step_size = float(step_size)
        self.alpha = float(alpha)
        self.d_safe = float(d_safe)

        self.node_list = [self.q_start]
        self.dim = len(q_start)
        self.q_start.cost = 0.0
        self.ee_link_index = 2

    # Helpers to sync q with PyBullet

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
        """
        3 x n linear Jacobian for a sphere point in world frame.
        """
        q = np.asarray(q, dtype=float)
        # velocities are not used for Jacobian
        zeros = [0.0] * len(q)
        J_pos, J_orn = p.calculateJacobian(
            self.robot_id, link_index, local_pos, q.tolist(), zeros, zeros
        )
        return np.array(J_pos)  # shape (3, n)

    # SDF gradient in world frame

    def _sdf_grad_world(self, x, eps=1e-3):
        """
        Finite-difference gradient of SDF at world point x (3,).epsilon = 1e-3 for numerical stability (avoid division by zero) to compute the gradient.
        """
        x = np.asarray(x, dtype=float)
        grad = np.zeros(3)
        for i in range(3):
            e = np.zeros(3)
            e[i] = eps
            grad[i] = (self.env_sdf(x + e) - self.env_sdf(x - e)) / (2.0 * eps)
        return grad

    # h(q) and dh/dq for all spheres

    def sdf_and_grad(self, q):
        """
        Compute CBF values h_i(q) = d_i(q) - d_safe and their gradients dh_i/dq for all spheres.

        Returns:
        - h:  (m,) barrier values (one per sphere)
        - dh: (m, n) Jacobian of h wrt q for all spheres (m = number of spheres, n = number of joints)
        """
        q = np.asarray(q, dtype=float)
        self._set_robot_config(q)

        h_list = []
        dh_list = []

        for sph in self.spheres:
            link_index = sph["link_index"]
            local_pos = sph["local_pos"]

            # world position of sphere center
            x = self._sphere_world_pos(link_index, local_pos)

            # signed distance at that point
            d = self.env_sdf(x)
            h_i = d - self.d_safe

            # ∇φ(x) wrt x
            grad_phi = self._sdf_grad_world(x)  # (3,)

            # Jacobian of x wrt q
            J_pos = self._sphere_jacobian(link_index, local_pos, q)  # (3, n)

            # dh_i/dq = ∇φ(x)^T J_pos  -> (n,)
            dh_i = grad_phi @ J_pos

            h_list.append(h_i)
            dh_list.append(dh_i)

        h = np.array(h_list)  # (m,)
        dh = np.vstack(dh_list)  # (m, n)
        return h, dh

    def _project_onto_cbf_constraints(self, u_des, h, dh, max_iters=10):
        """
        Project the desired velocity u_des onto the CBF constraints for all spheres.
        """
        u = u_des.copy()
        m = h.shape[0]
        for _ in range(max_iters):
            changed = False
            for i in range(m):
                # Enforce constraint whenever h[i] <= d_safe_threshold (or simply always)
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

    def step(self, q_from, q_to, num_substeps=10):
        """
        Step from q_from to q_to, that should
        (a) return the q_to if it is within the self.step_size or
        (b) only step so far as self.step_size, returning the new node within that distance
        """
        q = np.asarray(q_from, dtype=float)
        q_to = np.asarray(q_to, dtype=float)

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

    def plan(self):
        """
        Basic RRT with CBF-constrained steering (self.step) instead of discrete collision checking.
        Returns an NxD array of joint configs from start to goal, or None if no path is found.
        """
        for _ in range(self.max_iter):
            # Sample goal with some probability, otherwise sample uniformly
            if np.random.rand() < 0.10:
                q_rand = self.q_goal.joint_angles.copy()
            else:
                q_rand = np.array(
                    [np.random.uniform(lo, hi) for (lo, hi) in self.q_limits],
                    dtype=float,
                )

            # Reject samples that are already in the unsafe set
            if check_node_collision(
                self.robot_id,
                self.obstacle_ids,
                self.joint_indices,
                q_rand,
                distance=0.0,  # or self.d_safe
            ):
                continue  # reject this sample and draw a new one

            # Nearest node in joint space
            nearest = self.get_nearest_node(q_rand)

            # CBF-steered step from nearest toward q_rand
            q_new = self.step(nearest.joint_angles, q_rand)

            # If we couldn't move (e.g. CBF blocked), skip
            if np.allclose(q_new, nearest.joint_angles):
                continue

            # Reject new node if it ends up inside unsafe set
            if check_node_collision(
                self.robot_id,
                self.obstacle_ids,
                self.joint_indices,
                q_new,
                distance=0.0,  # or self.d_safe
            ):
                continue

            # Reject new node if the edge is colliding
            if check_edge_collision(
                self.robot_id, 
                self.obstacle_ids,
                nearest.joint_angles,
                q_new,
            ):
                continue

            # Add new node to the tree
            new_node = Node(q_new)
            new_node.parent = nearest
            self.node_list.append(new_node)
            plot_rrt_edge(
                robot_id=self.robot_id,
                q_from=nearest.joint_angles,
                q_to=new_node.joint_angles,
                ee_link_index=self.ee_link_index,
                line_color=[0, 1, 0],
                line_width=1,
                duration=0,
            )

            # Try to connect to goal if close enough
            if np.linalg.norm(new_node.joint_angles - self.q_goal.joint_angles) <= self.step_size:
                q_goal_proj = self.step(new_node.joint_angles, self.q_goal.joint_angles)

                # Optional: also check that the projected goal config is safe
                if check_node_collision(
                    self.robot_id,
                    self.obstacle_ids,
                    self.joint_indices,
                    q_goal_proj,
                    distance=0.0,  # or self.d_safe
                ):
                    continue

                # If CBF allows us to reach (or nearly reach) the goal config
                if np.linalg.norm(q_goal_proj - self.q_goal.joint_angles) < self.step_size:
                    # Make an actual goal node that lives in the tree
                    goal_node = Node(q_goal_proj)
                    goal_node.parent = new_node
                    self.node_list.append(goal_node)

                    # Draw the final edge in the visualization
                    plot_rrt_edge(
                        robot_id=self.robot_id,
                        q_from=new_node.joint_angles,
                        q_to=goal_node.joint_angles,
                        ee_link_index=self.ee_link_index,
                        line_color=[1, 0, 0],
                        line_width=2,
                        duration=0,
                    )

                    # Build path from this goal_node
                    path = []
                    node = goal_node
                    while node is not None:
                        path.append(np.asarray(node.joint_angles, dtype=float))
                        node = node.parent
                    path.reverse()
                    return np.vstack(path)

        # Failed to find a path within max_iter
        return None


    def get_nearest_node(self, q_rand):
        """Find the nearest node in the tree to q_rand."""
        dists = [np.linalg.norm(node.joint_angles - q_rand) for node in self.node_list]
        nearest_index = np.argmin(dists)
        return self.node_list[nearest_index]
