import numpy as np
from build123d import *

class Spiral(BaseLineObject):
    """
    Line Object: Spiral

    Create a planar spiral (Archimedean spiral).

    Args:
        start_radius (float): Starting radius of the spiral.
        end_radius (float): Ending radius of the spiral.
        nr_turns (float): Number of turns of the spiral.
        plane (Plane, optional): Plane of the spiral. Defaults to Plane.XY.
        mode (Mode, optional): Combination mode. Defaults to Mode.ADD.
    """

    _applies_to = [BuildLine._tag]

    def __init__(
        self,
        start_radius: float,
        end_radius: float,
        nr_turns: float,
        resolution_factor: float = 10,
        plane: Plane = Plane.XY,
        mode: Mode = Mode.ADD,
    ):
        context: BuildLine = BuildLine._get_context(self)

        theta_start = 0
        theta_end = nr_turns * 2 * np.pi
        num_points = int(nr_turns * resolution_factor)
        a = start_radius
        b = (end_radius - start_radius) / theta_end

        theta = np.linspace(theta_start, theta_end, num_points)
        radius = a + b * theta

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.zeros_like(theta)

        points = [Vector(x[i], y[i], z[i]) for i in range(num_points)]

        # If a different plane is specified, transform the points
        if plane != Plane.XY:
            points = [plane.from_local_coords(p) for p in points]

        spline = Edge.make_spline(points)
        super().__init__(spline, mode=mode)




if __name__ == "__main__":
    from ocp_vscode import *

    # Create a spiral
    spiral = Spiral(start_radius=1, end_radius=10, nr_turns=2)

    # Display the spiral
    show(spiral)
