# SDF test

import os
import numpy as np
import warp as wp

wp.init()

from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.data import Fielduint8
from pumpkin_pulse.operator import Operator
from pumpkin_pulse.operator.geometry import (
    SignedDistanceFunction,
)
from pumpkin_pulse.operator.saver import FieldSaver

if __name__ == '__main__':

    # Define simulation parameters
    dx = 0.01
    origin = (0.0, 0.0, 0.0)
    spacing = (dx, dx, dx)
    shape = (int(1.0/dx), int(1.0/dx), int(1.0/dx))
    nr_cells = shape[0] * shape[1] * shape[2]
    print(f"Number of cells: {nr_cells}")

    # Make sdf function
    sphere = SignedDistanceFunction.sphere(
        center=(0.5, 0.5, 0.5),
        radius=0.5,
    )
    cylinder = SignedDistanceFunction.cylinder(
        center=(0.5, 0.5, 0.5),
        radius=0.3,
        height=0.5,
    )
    cylinder_x = SignedDistanceFunction.rotate(
        cylinder,
        center=(0.5, 0.5, 0.5),
        centeraxis=(0, 0, 1.0),
        angle=np.pi/2.0,
    )
    cylinder_y = cylinder
    cylinder_z = SignedDistanceFunction.rotate(
        cylinder,
        center=(0.5, 0.5, 0.5),
        centeraxis=(1.0, 0, 0),
        angle=np.pi/2.0,
    )
    multi_cylinder = SignedDistanceFunction.union(
        SignedDistanceFunction.union(cylinder_x, cylinder_y),
        cylinder_z,
    )
    holed_sphere = SignedDistanceFunction.difference(
        sphere,
        multi_cylinder,
    )

    # Make the operators
    wp.clear_kernel_cache()
    sdf_operator = SignedDistanceFunction(holed_sphere)
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
    id_field = sdf_operator(id_field, 1)

    # Save the fields
    field_saver(
        {"id_field": id_field},
        "id_field.vtk",
    )
