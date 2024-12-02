from __future__ import annotations

import numpy as np
from build123d import *

class Helix(BaseLineObject):
    """
    Line Object: Custom Helix

    Create a 3D helix path by manually generating the points along the helix.

    Args:
        radius (float): Radius of the helix.
        pitch (float): The distance between each turn along the helix axis.
        height (float): Total height of the helix.
        nr_turns (float): Number of turns of the helix.
        resolution_factor (float): Number of points per turn (higher value for smoother helix).
        axis (VectorLike): Direction of the helix axis.
        start_angle (float): Starting angle in degrees (default is 0).
        mode (Mode, optional): Combination mode. Defaults to Mode.ADD.
    """

    _applies_to = [BuildLine._tag]

    def __init__(
        self,
        radius: float,
        pitch: float,
        height: float = None,
        nr_turns: float = None,
        resolution_factor: float = 10,
        start_angle: float = 0.0,
        mode: Mode = Mode.ADD,
    ):
        context: BuildLine = BuildLine._get_context(self)

        # Calculate the number of turns if height is provided
        if height is not None:
            nr_turns = height / pitch
        elif nr_turns is not None:
            height = nr_turns * pitch
        else:
            raise ValueError("Either height or nr_turns must be specified.")

        # Number of points for the given resolution
        num_points = int(nr_turns * resolution_factor * 36)  # 36 points per turn

        # Convert start_angle to radians
        start_angle_rad = np.deg2rad(start_angle)

        # Generate theta values
        theta = np.linspace(start_angle_rad, start_angle_rad + nr_turns * 2 * np.pi, num_points)

        # Calculate the coordinates
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = (pitch / (2 * np.pi)) * theta

        # Make the points
        points = [Vector(x[i], y[i], z[i]) for i in range(num_points)]

        spline = Edge.make_spline(points)
        super().__init__(spline, mode=mode)

if __name__ == "__main__":
    from ocp_vscode import *

    # Create a custom helix
    custom_helix = Helix(
        radius=1.0,
        pitch=0.5,
        height=5.0,
        resolution_factor=10,
        start_angle=0.0,
    )

    # Display the custom helix
    show(custom_helix)

