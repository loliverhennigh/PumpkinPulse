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
    Coil,
)
from pumpkin_pulse.operator.saver import FieldSaver

if __name__ == '__main__':

    # Define simulation parameters
    dx = 0.0025
    origin = (0.0, 0.0, 0.0)
    spacing = (dx, dx, dx)
    shape = (int(1.0/dx), int(1.0/dx), int(1.0/dx))
    nr_cells = shape[0] * shape[1] * shape[2]
    print(f"Number of cells: {nr_cells}")

    # Capacitor parameters
    coil_radius = 0.3
    cable_radius = 0.01
    insulator_thickness = 0.01
    nr_turns_z = 5
    nr_turns_r = 5
    center = (0.5, 0.5, 0.5)

    # Make the operators
    coil_operator = Coil(
        coil_radius=coil_radius,
        cable_radius=cable_radius,
        insulator_thickness=insulator_thickness,
        nr_turns_z=nr_turns_z,
        nr_turns_r=nr_turns_r,
        center=center,
        conductor_id=1,
        insulator_id=2,
    )
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
    id_field = coil_operator(id_field)

    # Save the fields
    field_saver(
        {"id_field": id_field},
        "id_field.vtk",
    )
