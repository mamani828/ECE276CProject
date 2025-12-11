import numpy as np
from utils import (
    get_ee_position,
    plot_link_coordinate_frames,
    check_edge_collision,
    plot_rrt_edge,
)


class Node:
    def __init__(self, joint_angles):
        self.joint_angles = np.array(
            joint_angles
        )  # joint angles of the node in n-dimensional space
        self.parent = None


class RRT:
    def __init__(self, joint_angles):
        self.joint_angles = np.array(
            joint_angles
        )  # joint angles of the node in n-dimensional space
        self.parent = None
        self.cost = 0.0  # Cost to reach this node from the start node for RRT*


class RRT:
    def __init__(
        self,
        q_start,
        q_goal,
        robot_id,
        obstacle_ids,
        q_limits,
        max_iter=10000,
        step_size=0.5,
    ):
        """
        RRT Initialization.

        Parameters:
        - q_start: List of starting joint angles [x1, x2, ..., xn].
        - q_goal: List of goal joint angles [x1, x2, ..., xn].
        - obstacle_ids: List of obstacles, each as a tuple ([center1, center2, ..., centern], radius).
        - q_limits: List of tuples [(min_x1, max_x1), ..., (min_xn, max_xn)] representing the limits in each dimension.
        - max_iter: Maximum number of iterations.
        - step_size: Maximum step size to expand the tree.
        """
        self.q_start = Node(q_start)
        self.q_goal = Node(q_goal)
        self.obstacle_ids = obstacle_ids
        self.robot_id = robot_id
        self.q_limits = q_limits
        self.max_iter = max_iter
        self.step_size = step_size
        self.node_list = [self.q_start]

        self.dim = len(q_start)
        self.q_start.cost = 0.0
        self.ee_link_index = 2

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

    def update_descendant_costs(self, root):
        stack = [root]
        while stack:
            current = stack.pop()
            for child in self.node_list:
                if child.parent is current:
                    child.cost = current.cost + self.edge_cost(
                        current.joint_angles, child.joint_angles
                    )
                    stack.append(child)

    def get_near_nodes(self, new_node, radius):
        center = np.asarray(new_node.joint_angles, dtype=float)
        near = []
        for node in self.node_list:
            if np.linalg.norm(node.joint_angles - center) <= radius:
                near.append(node)
        return near

    def plan(self):
        """Run the RRT algorithm to find a path of dimension Nx3. Limit the search
        to only max_iter iterations."""
        # Early collision checks
        for _ in range(self.max_iter):
            if np.random.rand() < 0.10:  # get sample
                q_rand = self.q_goal.joint_angles.copy()
            else:
                q_rand = np.array(
                    [np.random.uniform(lo, hi) for (lo, hi) in self.q_limits],
                    dtype=float,
                )
            #TODO add check node collision here
            nearest = self.get_nearest_node(q_rand)

            # Steer from nearest toward sample, capped by self.step_size
            q_new = self.step(nearest.joint_angles, q_rand)
            if check_edge_collision(
                self.robot_id,
                self.obstacle_ids,
                nearest.joint_angles.tolist(),
                q_new.tolist(),
                discretization_step=0.01,
            ):
                continue
            new_node = Node(q_new)
            new_node.parent = nearest
            self.node_list.append(new_node)
            plot_rrt_edge(
                robot_id=self.robot_id,
                q_from=nearest.joint_angles,
                q_to=new_node.joint_angles,
                ee_link_index=self.ee_link_index,
                line_color=[0, 1, 0],  # green tree
                line_width=1,
                duration=0,
            )
            if (
                np.linalg.norm(new_node.joint_angles - self.q_goal.joint_angles)
                <= self.step_size
            ):
                if not check_edge_collision(
                    self.robot_id,
                    self.obstacle_ids,
                    new_node.joint_angles.tolist(),
                    self.q_goal.joint_angles.tolist(),
                    discretization_step=0.01,
                ):
                    self.q_goal.parent = new_node
                    arm_id = self.robot_id
                    plot_link_coordinate_frames(
                        arm_id, [0, 1, 2], axis_length=0.1, duration=0
                    )  # Visualize the robot coordinate frames
                    path = []
                    node = self.q_goal
                    while node is not None:
                        path.append(np.asarray(node.joint_angles, dtype=float))
                        node = node.parent
                    path.reverse()
                    return path
        return None

    def edge_cost(self, q1, q2):
        """Cartesian edge cost: EE distance between two configs."""
        from utils import get_ee_position

        x1 = get_ee_position(self.robot_id, q1, self.ee_link_index)
        x2 = get_ee_position(self.robot_id, q2, self.ee_link_index)
        return np.linalg.norm(x2 - x1)

    def plan2(self):
        """Run the RRT algorithm to find a path of dimension Nx3. Limit the search to only max_iter iterations."""
        best_goal_node = None
        best_goal_cost = float("inf")
        for _ in range(self.max_iter):
            if np.random.rand() < 0.10:
                q_rand = self.q_goal.joint_angles.copy()
            else:
                q_rand = np.array(
                    [np.random.uniform(lo, hi) for (lo, hi) in self.q_limits],
                    dtype=float,
                )
            nearest = self.get_nearest_node(q_rand)
            q_new = self.step(nearest.joint_angles, q_rand)
            if check_edge_collision(
                self.robot_id,
                self.obstacle_ids,
                nearest.joint_angles.tolist(),
                q_new.tolist(),
                discretization_step=0.01,
            ):
                continue

            new_node = Node(q_new)
            radius = 0.5
            near_nodes = self.get_near_nodes(new_node, radius)
            parent = nearest
            min_cost = nearest.cost + self.edge_cost(
                nearest.joint_angles, new_node.joint_angles
            )

            for node in near_nodes:
                if check_edge_collision(
                    self.robot_id,
                    self.obstacle_ids,
                    node.joint_angles.tolist(),
                    new_node.joint_angles.tolist(),
                    discretization_step=0.01,
                ):
                    continue
                cost_through_node = node.cost + self.edge_cost(
                    node.joint_angles, new_node.joint_angles
                )
                if cost_through_node < min_cost:
                    parent = node
                    min_cost = cost_through_node

            new_node.parent = parent
            new_node.cost = min_cost
            self.node_list.append(new_node)
            for node in near_nodes:
                if node is parent:
                    continue
                cost_through_new = new_node.cost + self.edge_cost(
                    new_node.joint_angles, node.joint_angles
                )

                if cost_through_new + 1e-9 < node.cost:  # tiebreaker

                    if not check_edge_collision(
                        self.robot_id,
                        self.obstacle_ids,
                        new_node.joint_angles.tolist(),
                        node.joint_angles.tolist(),
                        discretization_step=0.01,
                    ):
                        node.parent = new_node
                        node.cost = cost_through_new
                        self.update_descendant_costs(node)
            if (
                np.linalg.norm(new_node.joint_angles - self.q_goal.joint_angles)
                <= self.step_size
            ):
                if not check_edge_collision(
                    self.robot_id,
                    self.obstacle_ids,
                    new_node.joint_angles.tolist(),
                    self.q_goal.joint_angles.tolist(),
                    discretization_step=0.01,
                ):
                    goal_cost_candidate = new_node.cost + self.edge_cost(
                        new_node.joint_angles, self.q_goal.joint_angles
                    )
                    if goal_cost_candidate < best_goal_cost:
                        self.q_goal.parent = new_node
                        self.q_goal.cost = goal_cost_candidate
                        best_goal_cost = goal_cost_candidate
                        best_goal_node = self.q_goal
        if best_goal_node is None:
            return None
        path = []
        node = best_goal_node
        while node is not None:
            plot_link_coordinate_frames(arm_id, [0, 1, 2], axis_length=0.1, duration=0)
            path.append(np.asarray(node.joint_angles, dtype=float))
            node = node.parent
        path.reverse()
        return np.vstack(path)
