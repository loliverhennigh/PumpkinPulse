# Electromagnet geometry test

import os
import numpy as np
import warp as wp
import matplotlib.pyplot as plt
import dataclasses

wp.init()

from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.data import Fielduint8
from pumpkin_pulse.operator import Operator
from pumpkin_pulse.operator.geometry.circuit import (
    Capacitor,
    Coil,
    Element,
    Cable,
)
from pumpkin_pulse.operator.saver import FieldSaver

def make_circuit(
    coil_radius,
    cable_radius,
    insulator_thickness,
    nr_turns_z,
    nr_turns_r,
    center,
    conductor_id,
    insulator_id,
    resistor_id,
    dielectric_id,
    switch_id,
):

    # Store the operators in list
    operators = []

    # Make the coil
    coil = Coil(
        coil_radius=coil_radius,
        cable_radius=cable_radius,
        insulator_thickness=insulator_thickness,
        nr_turns_z=nr_turns_z,
        nr_turns_r=nr_turns_r,
        conductor_id=conductor_id,
        insulator_id=insulator_id,
        center=center,
        square=True,
    )

    # Make the resistor
    resistor = Element(
        cable_radius=cable_radius,
        insulator_thickness=insulator_thickness,
        element_length=3.0 * cable_radius,
        element_id=resistor_id,
        insulator_id=insulator_id,
        center=(
            center[0] - (coil_radius + 2.0 * (nr_turns_r + 1.0) * (cable_radius + insulator_thickness)),
            center[1] - 3.0 * cable_radius - insulator_thickness,
            (coil.output_point[2] + coil.input_point[2]) / 2.0,
        ),
    )

    # Make the switch
    switch = Element(
        cable_radius=cable_radius,
        insulator_thickness=insulator_thickness,
        element_length=3.0 * cable_radius,
        element_id=switch_id,
        insulator_id=insulator_id,
        center=(
            center[0] - (coil_radius + 2.0 * (nr_turns_r + 1.0) * (cable_radius + insulator_thickness)),
            center[1] + 3.0 * cable_radius + insulator_thickness,
            (coil.output_point[2] + coil.input_point[2]) / 2.0,
        ),
    )

    # Make the capacitor
    capacitor = Capacitor(
        capacitor_width=5.0 * cable_radius,
        conductor_plate_thickness=cable_radius,
        dielectric_thickness=cable_radius,
        cable_radius=cable_radius,
        insulator_thickness=insulator_thickness,
        conductor_id=conductor_id,
        dielectric_id=dielectric_id,
        insulator_id=insulator_id,
        center=(
            center[0] - (coil_radius + 2.0 * (nr_turns_r + 1.0) * (cable_radius + insulator_thickness)),
            center[1],
            (coil.output_point[2] + coil.input_point[2]) / 2.0,
        ),
    )

    # Make cable from switch to coil
    cable_0 = Cable(
        input_point=switch.output_point,
        input_normal=switch.output_normal,
        output_point=coil.output_point,
        output_normal=coil.output_normal,
        cable_radius=cable_radius,
        insulator_thickness=insulator_thickness,
        conductor_id=conductor_id,
        insulator_id=insulator_id,
        scale=1.35 * coil_radius,
    )

    # Make cable from resistor to coil
    cable_1 = Cable(
        input_point=resistor.input_point,
        input_normal=resistor.input_normal,
        output_point=coil.input_point,
        output_normal=coil.input_normal,
        cable_radius=cable_radius,
        insulator_thickness=insulator_thickness,
        conductor_id=conductor_id,
        insulator_id=insulator_id,
        scale=1.35 * coil_radius,
    )

    # Add the operators to the list
    operators.append(coil)
    operators.append(resistor)
    operators.append(switch)
    operators.append(capacitor)
    operators.append(cable_0)
    operators.append(cable_1)

    return operators

if __name__ == '__main__':

    # Define simulation parameters
    dx = 0.0025
    origin = (0.0, 0.0, 0.0)
    spacing = (dx, dx, dx)
    shape = (int(1.0/dx), int(1.0/dx), int(1.0/dx))
    nr_cells = shape[0] * shape[1] * shape[2]
    print(f"Number of cells: {nr_cells}")

    # Capacitor parameters
    coil_radius = 0.2
    cable_radius = 0.02
    insulator_thickness = 0.01
    nr_turns_z = 0
    nr_turns_r = 1
    center = (0.6, 0.5, 0.5)

    # Make the operators
    operators = make_circuit(
        coil_radius=coil_radius,
        cable_radius=cable_radius,
        insulator_thickness=insulator_thickness,
        nr_turns_z=nr_turns_z,
        nr_turns_r=nr_turns_r,
        center=center,
        conductor_id=1,
        insulator_id=2,
        resistor_id=3,
        dielectric_id=4,
        switch_id=5,
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
    for operator in operators:
        operator(id_field)

    # Save the fields
    field_saver(
        {"id_field": id_field},
        "id_field.vtk",
    )
