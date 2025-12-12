def get_env(env_name: str):
    if env_name == "simple":
        # collision_ids = []
        collision_positions = [
            [0.3, 0.5, 0.251],
            [-0.3, 0.3, 0.101],
            [-1, -0.15, 0.251],
            [-1, -0.15, 0.752],
            [-0.5, -1, 0.251],
            [0.5, -0.35, 0.201],
            [0.5, -0.35, 0.602],
        ]
        collision_orientations = [
            [0, 0, 0.5],
            [0, 0, 0.2],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0.25],
            [0, 0, 0.5],
        ]
        collision_scales = [0.5, 0.25, 0.5, 0.5, 0.5, 0.4, 0.4]

        # Having colorful cubes
        colors = [
            [1, 0, 0, 1],  # red
            [0, 1, 0, 1],  # green
            [0, 0, 1, 1],  # blue
            [1, 1, 0, 1],  # yellow
            [1, 0, 1, 1],  # magenta
            [0, 1, 1, 1],  # cyan
            [0.8, 0.4, 0, 1],  # orange
        ]
    if env_name == "complex":
        # collision_ids = []
        collision_positions = [
            [0.3, 0.5, 0.251],
            [-0.3, 0.3, 0.101],
            [-1, -0.15, 0.251],
            # [-1, -0.15, 0.752],
            [-0.5, -1, 0.251],
            # [-0.5, -1, 0.752],
            [0.5, -0.35, 0.201],
            [0.5, -0.35, 0.602],
            [-0.75, -0.575, 1.003],  # Top cube
        ]
        collision_orientations = [
            [0, 0, 0.5],
            [0, 0, 0.2],
            [0, 0, 0],
            # [0, 0, 1],
            [0, 0, 0],
            # [0, 0, 0],
            [0, 0, 0.25],
            [0, 0, 0.5],
            [0, 0, 0],
        ]
        collision_scales = [0.5, 0.25, 0.5, 0.5, 0.4, 0.4, 1.0]

        # Having colorful cubes
        colors = [
            [1, 0, 0, 1],  # red
            [0, 1, 0, 1],  # green
            [0, 0, 1, 1],  # blue
            # [1, 1, 0, 1],   # yellow
            [1, 0, 1, 1],  # magenta
            # [0, 0, 0, 1],   # black
            [0, 1, 1, 1],  # cyan
            [0.8, 0.4, 0, 1],  # orange
            [0, 0.5, 0.5, 1],  # teal
        ]
    return collision_positions, collision_orientations, collision_scales, colors
