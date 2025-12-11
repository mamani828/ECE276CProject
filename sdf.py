import math

import pybullet as p
import numpy as np

from matplotlib import pyplot as plt


def make_pybullet_env_sdf(obstacle_ids, max_distance=5.0, probe_radius=1e-3):
    """
    Returns env_sdf(point) that gives signed distance to the closest collider
    among `obstacle_ids`, using PyBullet getClosestPoints.

    Args:
        obstacle_ids (list): List of obstacle IDs in PyBullet.
        max_distance (float): The maximum distance to search for obstacles.
        probe_radius (float): The radius of the probe sphere.
    Returns:
        function: The SDF function that gives the signed distance to the closest obstacle.
    """

    # Create a tiny probe sphere we will move around for queries
    probe_col = p.createCollisionShape(p.GEOM_SPHERE, radius=probe_radius)
    probe_body = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=probe_col,
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )

    def env_sdf(point):
        """
        Ensure point is a plain list/tuple and move probe to query point.
        Args:
            point (list/tuple): The point to query the SDF at.
        Returns:
            float: The signed distance to the closest obstacle.
        """
        point = list(point)

        # Move probe to query point
        p.resetBasePositionAndOrientation(
            bodyUniqueId=probe_body, posObj=point, ornObj=[0.0, 0.0, 0.0, 1.0]
        )

        d_min = max_distance
        found_any = False

        for obs_id in obstacle_ids:
            # Query closest points within a given search radius
            cps = p.getClosestPoints(
                bodyA=probe_body, bodyB=obs_id, distance=max_distance
            )

            # If no closest points are found, continue to the next obstacle
            if cps is None:
                continue

            for cp in cps:
                # cp[8] is contactDistance:
                #   > 0 when separated
                #   = 0 when touching
                #   < 0 when penetrating
                contact_dist = cp[8]

                # Distance from probe center to obstacle surface:
                #   we add probe_radius because Bullet uses surfaces of shapes
                d = contact_dist + probe_radius

                if (not found_any) or (d < d_min):
                    d_min = d
                    found_any = True

        if not found_any:
            # No obstacle within max_distance
            return max_distance

        return d_min

    return env_sdf


def visualize_sdf_slices(
    env_sdf,
    heights,
    x_range=(-3.0, 3.0),
    y_range=(-3.0, 3.0),
    resolution=0.1,
    cmap="seismic",
    vmin=-1.0,
    vmax=1.0,
):
    """
    Visualizes multiple 2D slices of the SDF at given heights in a single plot.
    
    Args:
        env_sdf (function): The SDF function.
        heights (list[float]): A list of z-heights to slice.
        x_range, y_range, resolution, cmap, vmin, vmax: Visualization parameters.
    """
    num_plots = len(heights)
    
    # Dynamically calculate grid size (e.g., max 3 columns)
    cols = 3
    rows = math.ceil(num_plots / cols)
    
    # Create the subplots
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
    axes = axes.flatten() if num_plots > 1 else [axes] # Ensure axes is iterable

    x_vals = np.arange(x_range[0], x_range[1], resolution)
    y_vals = np.arange(y_range[0], y_range[1], resolution)
    
    # Loop through heights and plot on corresponding axis
    for i, h in enumerate(heights):
        ax = axes[i]
        
        # Calculate SDF slice
        sdf_values = np.zeros((len(y_vals), len(x_vals)))
        for ix, x in enumerate(x_vals):
            for iy, y in enumerate(y_vals):
                point = (x, y, h)
                sdf_values[iy, ix] = env_sdf(point)
        
        # Plot
        im = ax.imshow(
            sdf_values,
            extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Height z={h}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    # Hide any empty subplots (if len(heights) isn't a multiple of cols)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Add a shared colorbar
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, label="Signed Distance")
    plt.show()
