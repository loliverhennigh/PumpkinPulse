# Tube test

import os
import numpy as np
import warp as wp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Explicitly import this module

wp.init()

from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.data import Fielduint8
from pumpkin_pulse.operator import Operator
from pumpkin_pulse.operator.geometry import (
    BezierTube,
)
from pumpkin_pulse.operator.saver import FieldSaver

if __name__ == '__main__':

    # Define simulation parameters
    dx = 0.01
    origin = (-1.0, -1.0, -1.0)
    spacing = (dx, dx, dx)
    shape = (int(3.0/dx), int(3.0/dx), int(3.0/dx))
    nr_cells = shape[0] * shape[1] * shape[2]
    print(f"Number of cells: {nr_cells}")

    # Make helix path for tube
    point_0 = np.array([0.2, 0.2, 0.2])
    normal_0 = np.array([0.0, 0.0, 1.0])
    point_1 = np.array([0.8, 0.8, 0.8])
    normal_1 = np.array([0.0, 0.0, 1.0])

    # Make the operators
    bezier_tube = BezierTube()
    field_saver = FieldSaver()

    # Make the constructor
    constructor = Constructor(
        shape=shape,
        origin=origin,
        spacing=spacing,
    )

    # Make the fields
    id_field = constructor.create_field(
        dtype=wp.uint8,
        cardinality=1,
    )

    # Initialize the fields
    id_field = bezier_tube(
        id_field,
        point_0,
        normal_0,
        point_1,
        normal_1,
        1000,
        1.0,
        0.05,
        1,
    )

    # Save the fields
    field_saver(
        {"id_field": id_field},
        "id_field.vtk",
    )
