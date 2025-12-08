import pybullet as p
import numpy as np
from scipy.stats import norm
from utils import plot_rrt_edge

class Node:
    def __init__(self, joint_angles, covariance=None):
        self.joint_angles = np.array(joint_angles)
        # Initialize covariance (uncertainty matrix)
        if covariance is None:
            self.cov = np.zeros((len(joint_angles), len(joint_angles)))
        else:
            self.cov = np.array(covariance)
        self.parent = None
        self.cost = 0.0

class RRT_CBF_Stochastic:
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
        confidence_level=0.95, # New: e.g., 95% confidence of no collision
        process_noise_std=0.01 # New: Assumed process noise per step
    ):
        # Initial uncertainty (can be zero or small value)
        init_cov = np.eye(len(q_start)) * 1e-4
        
        self.q_start = Node(q_start, init_cov)
        self.q_goal = Node(q_goal) # Goal uncertainty doesn't strictly matter for planning target
        
        self.robot_id = robot_id
        self.obstacle_ids = obstacle_ids
        self.q_limits = np.array(q_limits, dtype=float)
        self.env_sdf = env_sdf
        self.spheres = spheres
        self.joint_indices = joint_indices
        self.dt = 0.05
        self.u_max = 1.0
        self.max_iter = max_iter
        self.step_size = float(step_size)
        self.alpha = float(alpha)
        self.d_safe = float(d_safe)

        # Process Noise Matrix (Q)
        # We assume simple additive Gaussian noise on velocity/position integration
        self.Q = np.eye(len(q_start)) * (process_noise_std ** 2)
        
        # Chance Constraint Buffer
        # Inverse CDF (probit) for the confidence level (e.g., 1.645 for 95%)
        self.confidence_score = norm.ppf(confidence_level)

        self.node_list = [self.q_start]
        self.dim = len(q_start)
        self.ee_link_index = 2

    def _set_robot_config(self, q):
        q = np.asarray(q, dtype=float)
        for i, j_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, j_idx, q[i])

    def _sphere_world_pos(self, link_index, local_pos):
        link_state = p.getLinkState(self.robot_id, link_index, computeForwardKinematics=True)
        world_pos, _ = p.multiplyTransforms(link_state[4], link_state[5], local_pos, [0,0,0,1])
        return np.array(world_pos)

    def _sphere_jacobian(self, link_index, local_pos, q):
        q = np.asarray(q, dtype=float)
        zeros = [0.0] * len(q)
        J_pos, _ = p.calculateJacobian(self.robot_id, link_index, local_pos, q.tolist(), zeros, zeros)
        return np.array(J_pos)

    def _sdf_grad_world(self, x, eps=1e-3):
        x = np.asarray(x, dtype=float)
        grad = np.zeros(3)
        for i in range(3):
            e = np.zeros(3)
            e[i] = eps
            grad[i] = (self.env_sdf(x + e) - self.env_sdf(x - e)) / (2.0 * eps)
        return grad

    def sdf_and_grad(self, q):
        """Returns h(q) and Jacobian dh/dq"""
        q = np.asarray(q, dtype=float)
        self._set_robot_config(q)
        h_list = []
        dh_list = []

        for sph in self.spheres:
            link_idx = sph["link_index"]
            local_pos = sph["local_pos"]
            x = self._sphere_world_pos(link_idx, local_pos)
            
            # CBF Value
            d = self.env_sdf(x)
            h_i = d - self.d_safe
            
            # Gradients
            grad_phi = self._sdf_grad_world(x) # (3,)
            J_pos = self._sphere_jacobian(link_idx, local_pos, q) # (3, n)
            dh_i = grad_phi @ J_pos # (n,)

            h_list.append(h_i)
            dh_list.append(dh_i)

        return np.array(h_list), np.vstack(dh_list)

    def _project_onto_scbf_constraints(self, u_des, h, dh, sigma_q, max_iters=10):
        """
        Stochastic CBF Projection (QP Solver).
        Constraint: L_f h + L_g h u + alpha * h >= Confidence_Buffer
        Buffer = probit(p) * sqrt(dh^T * Sigma * dh)
        """
        u = u_des.copy()
        m = h.shape[0]
        
        for _ in range(max_iters):
            changed = False
            for i in range(m):
                # 1. Compute the Safety Buffer based on Uncertainty
                # Variance of the barrier h_i due to joint uncertainty Sigma_q
                # Var(h_i) approx (dh/dq)^T * Sigma_q * (dh/dq)
                grad_h = dh[i]
                h_variance = grad_h @ sigma_q @ grad_h.T
                h_std = np.sqrt(h_variance + 1e-12) 
                
                # The buffer pushes the robot further away if uncertainty is high
                safety_buffer = self.confidence_score * h_std

                # 2. Define the linear constraint: a^T u >= b
                # Standard: dh * u >= -alpha * h
                # Stochastic: dh * u >= -alpha * h + safety_buffer
                # We rearrange to: dh * u >= b_safe
                
                b_safe = -self.alpha * h[i] + safety_buffer
                
                # 3. Check violation
                val = grad_h @ u
                if val < b_safe:
                    # Project u onto the half-plane defined by the normal grad_h
                    denom = np.dot(grad_h, grad_h) + 1e-10
                    u = u + (b_safe - val) / denom * grad_h
                    changed = True
            if not changed:
                break
        return u

    def step(self, node_from, q_to_target, num_substeps=5):
        """
        Steers from node_from (config + covariance) toward q_to_target.
        Propagates uncertainty and applies SCBF.
        Returns: q_final, sigma_final
        """
        q_curr = node_from.joint_angles.copy()
        sigma_curr = node_from.cov.copy()
        q_target = np.asarray(q_to_target, dtype=float)

        reached_target = False

        for _ in range(num_substeps):
            direction = q_target - q_curr
            dist = np.linalg.norm(direction)
            if dist < 1e-3:
                reached_target = True
                break

            direction /= dist
            u_des = self.u_max * direction

            # Get Barriers and Gradients
            h, dh = self.sdf_and_grad(q_curr)
            
            # Project velocity using Stochastic CBF (accounts for sigma_curr)
            u_safe = self._project_onto_scbf_constraints(u_des, h, dh, sigma_curr)

            if np.linalg.norm(u_safe) < 1e-6:
                break # Blocked by uncertainty or obstacle

            # Propagate State (Euler Integration)
            q_curr = q_curr + u_safe * self.dt
            
            # Propagate Uncertainty (Linear Gaussian Assumption)
            # Sigma_k+1 = A Sigma_k A^T + Q
            # Here A (Jacobian of dynamics) is Identity for simple integrators
            sigma_curr = sigma_curr + self.Q * self.dt 

            q_curr = np.clip(q_curr, self.q_limits[:, 0], self.q_limits[:, 1])

        return q_curr, sigma_curr

    def plan(self):
        for _ in range(self.max_iter):
            # Sample
            if np.random.rand() < 0.20:
                q_rand = self.q_goal.joint_angles.copy()
            else:
                q_rand = np.array([np.random.uniform(lo, hi) for (lo, hi) in self.q_limits])

            # Nearest
            nearest_node = self.get_nearest_node(q_rand)

            # Steer (with Uncertainty Propagation)
            q_new, sigma_new = self.step(nearest_node, q_rand)

            if np.allclose(q_new, nearest_node.joint_angles):
                continue

            # Add Node
            new_node = Node(q_new, sigma_new)
            new_node.parent = nearest_node
            self.node_list.append(new_node)
            
            # Optional: Visualize edges (Standard RRT visualization)
            plot_rrt_edge(self.robot_id, nearest_node.joint_angles, new_node.joint_angles, self.ee_link_index)

            # Goal Check
            if np.linalg.norm(new_node.joint_angles - self.q_goal.joint_angles) <= self.step_size:
                # Try to reach goal exactly
                q_final, sigma_final = self.step(new_node, self.q_goal.joint_angles)
                if np.linalg.norm(q_final - self.q_goal.joint_angles) < self.step_size:
                    self.q_goal.parent = new_node
                    self.q_goal.cov = sigma_final
                    
                    # Backtrack
                    path = []
                    node = self.q_goal
                    while node is not None:
                        path.append(np.asarray(node.joint_angles, dtype=float))
                        node = node.parent
                    path.reverse()
                    return np.vstack(path)

        return None

    def get_nearest_node(self, q_rand):
        # Using Euclidean distance for selection (simplest for now)
        # Advanced: Use Mahalanobis distance d = sqrt((x-mu)T Sigma^-1 (x-mu))
        dists = [np.linalg.norm(node.joint_angles - q_rand) for node in self.node_list]
        return self.node_list[np.argmin(dists)]