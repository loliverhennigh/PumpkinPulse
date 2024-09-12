# Simple example for setting base parameters for the solver

import warp as wp
import numpy as np
from build123d import Rectangle, extrude, Location, Color
wp.init()

from pumpkin_pulse.solid import Solid
from pumpkin_pulse.material import Material, COPPER
from pumpkin_pulse.solver import Solver


if __name__ == '__main__':

    # Make Solid
    rect = Rectangle(1.0, 1.0)
    rect = extrude(rect, amount=1.0)
    solid = Solid(rect, COPPER, "Copper")

    # Make solver
    solver = Solver(
        solids=[solid],
        checkpoint_dir="checkpoint",
        spacing=(0.01, 0.01, 0.01),
        origin=(-2.0, -2.0, -2.0),
        shape=(400, 400, 400),
    )

    # Solve the problem
    solver.set_id_field()

    # Save the solution
    solver.save_fields(["id_field"])
