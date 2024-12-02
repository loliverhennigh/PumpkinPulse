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
    Tube,
)
from pumpkin_pulse.operator.saver import FieldSaver

if __name__ == '__main__':

    # Define simulation parameters
    dx = 0.005
    origin = (0.0, 0.0, 0.0)
    spacing = (dx, dx, dx)
    shape = (int(1.0/dx), int(1.0/dx), int(1.0/dx))
    nr_cells = shape[0] * shape[1] * shape[2]
    print(f"Number of cells: {nr_cells}")

    # Make helix path for tube
    p = np.linspace(0.0, 1.0, 1000)
    x = 0.5 + 0.2 * np.cos(20.0 * np.pi * p)
    y = 0.5 + 0.2 * np.sin(20.0 * np.pi * p)
    z = 0.25 + 0.5 * p
    path = np.array([x, y, z]).T
    outer_radius = np.ones(path.shape[0]) * 0.02
    inner_radius = np.ones(path.shape[0]) * 0.01

    # Make the operators
    tube = Tube()
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
    path = wp.from_numpy(path, dtype=wp.vec3)
    outer_radius = wp.from_numpy(outer_radius, dtype=wp.float32)
    inner_radius = wp.from_numpy(inner_radius, dtype=wp.float32)

    # Initialize the fields
    id_field = tube(
        id_field,
        path,
        outer_radius,
        1,
    )
    id_field = tube(
        id_field,
        path,
        inner_radius,
        2,
    )

    # Save the fields
    field_saver(
        {"id_field": id_field},
        "id_field.vtk",
    )
