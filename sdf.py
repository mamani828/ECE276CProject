import pybullet as p
import numpy as np
from matplotlib import pyplot as plt


def make_pybullet_env_sdf(obstacle_ids, max_distance=5.0, probe_radius=1e-3):
    """
    Returns env_sdf(point) that gives signed distance to the closest collider
    among `obstacle_ids`, using PyBullet getClosestPoints.

    point: (x, y, z)
    output:
        d < 0 : point is inside some obstacle (penetration depth)
        d = 0 : on the surface
        d > 0 : outside, distance to nearest surface (approximate, capped at max_distance)
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
        # Ensure point is a plain list/tuple
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


def visualize_sdf_slice(
    env_sdf,
    height=0.0,
    x_range=(-3.0, 3.0),
    y_range=(-3.0, 3.0),
    resolution=0.1,
    cmap="seismic",
    vmin=-1.0,
    vmax=1.0,
):
    """
    Visualizes a 2D slice of the SDF at a given height (z value).
    """

    import matplotlib.pyplot as plt

    x_vals = np.arange(x_range[0], x_range[1], resolution)
    y_vals = np.arange(y_range[0], y_range[1], resolution)

    sdf_values = np.zeros((len(y_vals), len(x_vals)))

    for ix, x in enumerate(x_vals):
        for iy, y in enumerate(y_vals):
            point = (x, y, height)
            sdf_values[iy, ix] = env_sdf(point)
    print(f"min sdf: {np.min(sdf_values)}, max sdf: {np.max(sdf_values)}")
    plt.imshow(
        sdf_values,
        extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label="Signed Distance")
    plt.title(f"SDF Slice at z={height}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
