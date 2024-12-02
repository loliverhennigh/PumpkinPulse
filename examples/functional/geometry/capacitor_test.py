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
    Capacitor,
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

    # Capacitor parameters
    capacitor_width = 0.5
    conductor_plate_thickness = 0.02
    dielectric_thickness = 0.05
    cable_radius = 0.02
    insulator_thickness = 0.02
    conductor_id = 1
    dielectric_id = 2
    insulator_id = 3
    center = (0.5, 0.5, 0.5)

    # Make the operators
    capacitor_operator = Capacitor(
        capacitor_width=capacitor_width,
        conductor_plate_thickness=conductor_plate_thickness,
        dielectric_thickness=dielectric_thickness,
        cable_radius=cable_radius,
        insulator_thickness=insulator_thickness,
        conductor_id=conductor_id,
        dielectric_id=dielectric_id,
        insulator_id=insulator_id,
        center=center,
        centeraxis=(1.0, 0.0, 0.0),
        angle=np.pi/2.0,
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
    id_field = capacitor_operator(id_field)

    # Save the fields
    field_saver(
        {"id_field": id_field},
        "id_field.vtk",
    )
