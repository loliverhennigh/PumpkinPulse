# Capacitor Circuit 

import os
import numpy as np
import warp as wp
import matplotlib.pyplot as plt
import dataclasses
import itertools
from tqdm import tqdm

wp.init()

from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.data import Fielduint8, Fieldfloat32
from pumpkin_pulse.operator import Operator
from pumpkin_pulse.operator.voxelize import (
    SignedDistanceFunction,
)
from pumpkin_pulse.operator.saver import FieldSaver

test = 1

class CapacitorCircuit(Operator):

    def __init__(
        self,
        coil_radius: float,
        cable_thickness_r: float,
        cable_thickness_y: float,
        insulator_thickness: float,
        dielectric_thickness: float,
        center: tuple = (0.0, 0.0, 0.0),
        centeraxis: tuple = (0.0, 1.0, 0.0),
        angle: float = 0.0,
        conductor_id: int = 1,
        insulator_id: int = 2,
        dielectric_id: int = 3,
        switch_id: int = 4,
        resistor_id: int = 5,
    ):

        # Store the parameters
        self.coil_radius = coil_radius
        self.cable_thickness_r = cable_thickness_r
        self.cable_thickness_y = cable_thickness_y
        self.insulator_thickness = insulator_thickness
        self.center = center
        self.centeraxis = centeraxis
        self.angle = angle
        self.conductor_id = conductor_id
        self.insulator_id = insulator_id
        self.dielectric_id = dielectric_id
        self.switch_id = switch_id
        self.resistor_id = resistor_id

        # Make conductor outline
        conductor_sdf = SignedDistanceFunction.cylinder(
            center=center,
            radius=coil_radius + cable_thickness_r,
            height=cable_thickness_y / 2.0,
        )
        conductor_sdf = SignedDistanceFunction.union(
            conductor_sdf,
            SignedDistanceFunction.box(
                center=(
                    center[0] + coil_radius + 2.0*cable_thickness_r,
                    center[1],
                    center[2],
                ),
                size=(
                    2.0 * cable_thickness_r,
                    cable_thickness_y / 2.0,
                    1.0 * cable_thickness_r + 0.5 * insulator_thickness,
                ),
            ),
        )
        conductor_sdf = SignedDistanceFunction.union(
            conductor_sdf,
            SignedDistanceFunction.box(
                center=(
                    center[0] + coil_radius + 5.5*cable_thickness_r,
                    center[1],
                    center[2],
                ),
                size=(
                    1.5 * cable_thickness_r,
                    cable_thickness_y / 2.0,
                    2.0 * cable_thickness_r + 0.5 * insulator_thickness + 0.5 * dielectric_thickness,
                ),
            ),
        )
        conductor_sdf = SignedDistanceFunction.difference(
            conductor_sdf,
            SignedDistanceFunction.cylinder(
                center=center,
                radius=coil_radius,
                height=cable_thickness_y / 2.0,
            ),
        )
        conductor_sdf = SignedDistanceFunction.difference(
            conductor_sdf,
            SignedDistanceFunction.box(
                center=(
                    center[0] + coil_radius + 2.5*cable_thickness_r,
                    center[1],
                    center[2],
                ),
                size=(
                    3.0 * cable_thickness_r,
                    cable_thickness_y / 2.0,
                    0.5 * insulator_thickness,
                ),
            ),
        )
        conductor_sdf = SignedDistanceFunction.difference(
            conductor_sdf,
            SignedDistanceFunction.box(
                center=(
                    center[0] + coil_radius + 5.5*cable_thickness_r,
                    center[1],
                    center[2],
                ),
                size=(
                    0.5 * cable_thickness_r,
                    cable_thickness_y / 2.0,
                    1.5 * cable_thickness_r + 0.5 * insulator_thickness,
                ),
            ),
        )
        conductor_sdf = SignedDistanceFunction.difference(
            conductor_sdf,
            SignedDistanceFunction.box(
                center=(
                    center[0] + coil_radius + 6.5*cable_thickness_r,
                    center[1],
                    center[2],
                ),
                size=(
                    0.5 * cable_thickness_r,
                    cable_thickness_y / 2.0,
                    0.5 * dielectric_thickness,
                ),
            ),
        )

        # Make switch elements
        switch_sdf_0 = SignedDistanceFunction.box(
            center=(
                center[0] + coil_radius + 3.5 * cable_thickness_r,
                center[1],
                center[2] + 0.5 * insulator_thickness + 0.5 * cable_thickness_r,
            ),
            size=(
                0.5 * cable_thickness_r,
                cable_thickness_y / 2.0,
                0.5 * cable_thickness_r,
            ),
        )
        switch_sdf_1 = SignedDistanceFunction.box(
            center=(
                center[0] + coil_radius + 3.5 * cable_thickness_r,
                center[1],
                center[2] - 0.5 * insulator_thickness - 0.5 * cable_thickness_r,
            ),
            size=(
                0.5 * cable_thickness_r,
                cable_thickness_y / 2.0,
                0.5 * cable_thickness_r,
            ),
        )

        # Make resistor element
        resistor_sdf = SignedDistanceFunction.box(
            center=(
                center[0] + coil_radius + 2.5 * cable_thickness_r,
                center[1],
                center[2] + 0.5 * insulator_thickness + 0.5 * cable_thickness_r,
            ),
            size=(
                0.5 * cable_thickness_r,
                cable_thickness_y / 2.0,
                0.5 * cable_thickness_r,
            ),
        )

        # Make dielectric element
        dielectric_sdf = SignedDistanceFunction.box(
            center=(
                center[0] + coil_radius + 6.5 * cable_thickness_r,
                center[1],
                center[2],
            ),
            size=(
                0.5 * cable_thickness_r,
                cable_thickness_y / 2.0,
                dielectric_thickness / 2.0,
            ),
        )

        # Rotate the elements
        conductor_sdf = SignedDistanceFunction.rotate(
            sdf=conductor_sdf,
            center=center,
            centeraxis=centeraxis,
            angle=angle,
        )
        switch_sdf_0 = SignedDistanceFunction.rotate(
            sdf=switch_sdf_0,
            center=center,
            centeraxis=centeraxis,
            angle=angle,
        )
        switch_sdf_1 = SignedDistanceFunction.rotate(
            sdf=switch_sdf_1,
            center=center,
            centeraxis=centeraxis,
            angle=angle,
        )
        resistor_sdf = SignedDistanceFunction.rotate(
            sdf=resistor_sdf,
            center=center,
            centeraxis=centeraxis,
            angle=angle,
        )
        dielectric_sdf = SignedDistanceFunction.rotate(
            sdf=dielectric_sdf,
            center=center,
            centeraxis=centeraxis,
            angle=angle,
        )

        # Make operator
        self.cunductor_operator = SignedDistanceFunction(sdf_func=conductor_sdf)
        self.switch_operator_0 = SignedDistanceFunction(sdf_func=switch_sdf_0)
        self.switch_operator_1 = SignedDistanceFunction(sdf_func=switch_sdf_1)
        self.resistor_operator = SignedDistanceFunction(sdf_func=resistor_sdf)
        self.dielectric_operator = SignedDistanceFunction(sdf_func=dielectric_sdf)

    @property
    def capacitor_surface_area(self):
        return self.cable_thickness_r * self.cable_thickness_y

    def __call__(
        self,
        id_field: Fielduint8
    ) -> Fielduint8:

        # Apply conductor operator
        id_field = self.cunductor_operator(id_field, self.conductor_id)
        id_field = self.switch_operator_0(id_field, self.switch_id)
        id_field = self.switch_operator_1(id_field, self.switch_id)
        id_field = self.resistor_operator(id_field, self.resistor_id)
        id_field = self.dielectric_operator(id_field, self.dielectric_id)

        return id_field


if __name__ == "__main__":

    # Define simulation parameters
    dx = 0.005 # 1 mm
    origin = (-2.0, -2.0, -2.0) # meters
    spacing = (dx, dx, dx)
    shape = (int(4.0 / dx), int(4.0 / dx), int(4.0 / dx))
    nr_cells = shape[0] * shape[1] * shape[2]
    print(f"Number of million cells: {nr_cells / 1e6}")

    # Define the coil parameters
    coil_radius = 0.2
    cable_thickness_r = 0.10
    cable_thickness_y = 0.05
    insulator_thickness = 0.05
    dielectric_thickness = 0.15

    # Make the field saver
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

    # Create the capacitor circuit
    circuit_operator = CapacitorCircuit(
        coil_radius=coil_radius,
        cable_thickness_r=cable_thickness_r,
        cable_thickness_y=cable_thickness_y,
        insulator_thickness=insulator_thickness,
        dielectric_thickness=dielectric_thickness,
    )

    # Run the Helion chamber
    id_field = circuit_operator(id_field)

    # Save the fields
    field_saver(
        {"id_field": id_field},
        "id_field.vtk",
    )

