# Chamber

import os
import numpy as np
import warp as wp
import matplotlib.pyplot as plt
import dataclasses
import itertools
from tqdm import tqdm
from typing import List

wp.init()

from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.data import Fielduint8, Fieldfloat32
from pumpkin_pulse.operator import Operator
from pumpkin_pulse.operator.voxelize import (
    Tube,
)
from pumpkin_pulse.operator.saver import FieldSaver

from geometry.chamber import Chamber
from geometry.circuit import CapacitorCircuit

class Reactor(Operator):
    def __init__(
        self,
        chamber_wall_thickness: float,
        interaction_radius: float,
        interaction_bounds: float,
        acceleration_bounds: float,
        formation_radius: float,
        formation_bounds: float,
        diverter_inlet_radius: float,
        diverter_inlet_bounds: float,
        diverter_radius: float,
        diverter_bounds: float,
        insulator_thickness: float,
        dielectric_thickness: float,
        vacuum_id: int,
        chamber_id: int,
        conductor_id: int,
        insulator_id: int,
        nr_interaction_coils: int,
        interaction_cable_thickness_r: float,
        interaction_cable_thickness_y: float,
        interaction_resistor_id: int,
        interaction_switch_id: List[int],
        interaction_dielectric_id: int,
        nr_acceleration_coils: int,
        accelerator_cable_thickness_r: float,
        accelerator_cable_thickness_y: float,
        accelerator_resistor_id: int,
        accelerator_switch_id: List[int],
        accelerator_dielectric_id: int,
        nr_formation_coils: int,
        formation_cable_thickness_r: float,
        formation_cable_thickness_y: float,
        formation_resistor_id: int,
        formation_switch_id: List[int],
        formation_dielectric_id: int,
        nr_diverter_inlet_coils: int,
        diverter_inlet_cable_thickness_r: float,
        diverter_inlet_cable_thickness_y: float,
        diverter_inlet_resistor_id: int,
        diverter_inlet_switch_id: List[int],
        diverter_inlet_dielectric_id: int,
    ):

        # Make the Chamber operator
        self.chamber_operator = Chamber(
            chamber_wall_thickness=chamber_wall_thickness,
            interaction_radius=interaction_radius,
            interaction_bounds=interaction_bounds,
            acceleration_bounds=acceleration_bounds,
            formation_radius=formation_radius,
            formation_bounds=formation_bounds,
            diverter_inlet_radius=diverter_inlet_radius,
            diverter_inlet_bounds=diverter_inlet_bounds,
            diverter_radius=diverter_radius,
            diverter_bounds=diverter_bounds,
            vacuum_id=vacuum_id,
            chamber_id=chamber_id,
        )

        # Make list to store all operators
        self.coil_operators = []

        # Interaction Coil parameters
        interaction_coil_radius = interaction_radius + chamber_wall_thickness
        interaction_positions = np.linspace(-interaction_bounds, interaction_bounds, nr_interaction_coils)
        for i in range(nr_interaction_coils):
            interaction_coil_operator = CapacitorCircuit(
                coil_radius=interaction_coil_radius,
                cable_thickness_r=interaction_cable_thickness_r,
                cable_thickness_y=interaction_cable_thickness_y,
                insulator_thickness=insulator_thickness,
                dielectric_thickness=dielectric_thickness,
                center=(interaction_positions[i], 0.0, 0.0),
                centeraxis=(0.0, 0.0, 1.0),
                angle=np.pi / 2,
                conductor_id=conductor_id,
                insulator_id=insulator_id,
                dielectric_id=interaction_dielectric_id,
                switch_id=interaction_switch_id[i],
                resistor_id=interaction_resistor_id,
            )
            self.coil_operators.append(interaction_coil_operator)

        # Acceleration Coil parameters
        accelerator_coil_posistions = np.linspace(
            interaction_bounds,
            acceleration_bounds,
            nr_acceleration_coils + 1,
            endpoint=False,
        )[1:]
        accelerator_coil_radius_slope = (formation_radius - interaction_radius) / (acceleration_bounds - interaction_bounds)
        accelerator_coil_radius = (
            interaction_radius
            + chamber_wall_thickness
            + accelerator_coil_radius_slope * (accelerator_coil_posistions - interaction_bounds)
        )
        for i in range(nr_acceleration_coils):
            lx_acceleration_coil_operator = CapacitorCircuit(
                coil_radius=float(accelerator_coil_radius[i]),
                cable_thickness_r=accelerator_cable_thickness_r,
                cable_thickness_y=accelerator_cable_thickness_y,
                insulator_thickness=insulator_thickness,
                dielectric_thickness=dielectric_thickness,
                center=(-float(accelerator_coil_posistions[i]), 0.0, 0.0),
                centeraxis=(0.0, 0.0, 1.0),
                angle=np.pi / 2,
                conductor_id=conductor_id,
                insulator_id=insulator_id,
                dielectric_id=accelerator_dielectric_id,
                switch_id=accelerator_switch_id[i],
                resistor_id=accelerator_resistor_id,
            )
            rx_acceleration_coil_operator = CapacitorCircuit(
                coil_radius=float(accelerator_coil_radius[i]),
                cable_thickness_r=accelerator_cable_thickness_r,
                cable_thickness_y=accelerator_cable_thickness_y,
                insulator_thickness=insulator_thickness,
                dielectric_thickness=dielectric_thickness,
                center=(float(accelerator_coil_posistions[i]), 0.0, 0.0),
                centeraxis=(0.0, 0.0, 1.0),
                angle=np.pi / 2,
                conductor_id=conductor_id,
                insulator_id=insulator_id,
                dielectric_id=accelerator_dielectric_id,
                switch_id=accelerator_switch_id[i],
                resistor_id=accelerator_resistor_id,
            )
            self.coil_operators.append(lx_acceleration_coil_operator)
            self.coil_operators.append(rx_acceleration_coil_operator)

        # Formation Coil parameters
        formation_coil_radius = formation_radius + chamber_wall_thickness
        formation_positions = np.linspace(
            acceleration_bounds,
            formation_bounds,
            nr_formation_coils
        )
        for i in range(nr_formation_coils):
            lx_formation_coil_operator = CapacitorCircuit(
                coil_radius=formation_coil_radius,
                cable_thickness_r=formation_cable_thickness_r,
                cable_thickness_y=formation_cable_thickness_y,
                insulator_thickness=insulator_thickness,
                dielectric_thickness=dielectric_thickness,
                center=(float(formation_positions[i]), 0.0, 0.0),
                centeraxis=(0.0, 0.0, 1.0),
                angle=np.pi / 2,
                conductor_id=conductor_id,
                insulator_id=insulator_id,
                dielectric_id=formation_dielectric_id,
                switch_id=formation_switch_id[i],
                resistor_id=formation_resistor_id,
            )
            rx_formation_coil_operator = CapacitorCircuit(
                coil_radius=formation_coil_radius,
                cable_thickness_r=formation_cable_thickness_r,
                cable_thickness_y=formation_cable_thickness_y,
                insulator_thickness=insulator_thickness,
                dielectric_thickness=dielectric_thickness,
                center=(-float(formation_positions[i]), 0.0, 0.0),
                centeraxis=(0.0, 0.0, 1.0),
                angle=np.pi / 2,
                conductor_id=conductor_id,
                insulator_id=insulator_id,
                dielectric_id=formation_dielectric_id,
                switch_id=formation_switch_id[i],
                resistor_id=formation_resistor_id,
            )
            self.coil_operators.append(lx_formation_coil_operator)
            self.coil_operators.append(rx_formation_coil_operator)

        # Inlet Diverter Coil parameters
        diverter_inlet_coil_radius = diverter_inlet_radius + chamber_wall_thickness
        diverter_inlet_positions = np.linspace(
            formation_bounds + chamber_wall_thickness + 1.0 * diverter_inlet_cable_thickness_y,
            diverter_inlet_bounds - chamber_wall_thickness - 1.0 * diverter_inlet_cable_thickness_y,
            nr_diverter_inlet_coils
        )
        for i in range(nr_diverter_inlet_coils):
            lx_diverter_inlet_coil_operator = CapacitorCircuit(
                coil_radius=diverter_inlet_coil_radius,
                cable_thickness_r=diverter_inlet_cable_thickness_r,
                cable_thickness_y=diverter_inlet_cable_thickness_y,
                insulator_thickness=insulator_thickness,
                dielectric_thickness=dielectric_thickness,
                center=(float(diverter_inlet_positions[i]), 0.0, 0.0),
                centeraxis=(0.0, 0.0, 1.0),
                angle=np.pi / 2,
                conductor_id=conductor_id,
                insulator_id=insulator_id,
                dielectric_id=diverter_inlet_dielectric_id,
                switch_id=diverter_inlet_switch_id[i],
                resistor_id=diverter_inlet_resistor_id,
            )
            rx_diverter_inlet_coil_operator = CapacitorCircuit(
                coil_radius=diverter_inlet_coil_radius,
                cable_thickness_r=diverter_inlet_cable_thickness_r,
                cable_thickness_y=diverter_inlet_cable_thickness_y,
                insulator_thickness=insulator_thickness,
                dielectric_thickness=dielectric_thickness,
                center=(-float(diverter_inlet_positions[i]), 0.0, 0.0),
                centeraxis=(0.0, 0.0, 1.0),
                angle=np.pi / 2,
                conductor_id=conductor_id,
                insulator_id=insulator_id,
                dielectric_id=diverter_inlet_dielectric_id,
                switch_id=diverter_inlet_switch_id[i],
                resistor_id=diverter_inlet_resistor_id,
            )
            self.coil_operators.append(lx_diverter_inlet_coil_operator)
            self.coil_operators.append(rx_diverter_inlet_coil_operator)

    def __call__(
        self,
        id_field: Fielduint8,
    ) -> Fielduint8:

        # Initialize id field
        id_field = self.chamber_operator(id_field)
        for coil_operator in self.coil_operators:
            id_field = coil_operator(id_field)

        return id_field



if __name__ == "__main__":

    # Define simulation parameters
    dx = 0.0005 # 1 mm
    origin = (-0.55, -0.085, -0.085) # meters
    spacing = (dx, dx, dx)
    shape = (int(1.1 / dx), int(0.17 / dx), int(0.17 / dx))
    nr_cells = shape[0] * shape[1] * shape[2]
    print(f"Number of million cells: {nr_cells / 1e6}")

    # Chamber parameters
    chamber_wall_thickness = 0.005
    interaction_radius = 0.01
    interaction_bounds = 0.1
    acceleration_bounds = 0.3
    formation_radius = 0.025
    formation_bounds = 0.4
    diverter_inlet_radius = 0.01
    diverter_inlet_bounds = 0.45
    diverter_radius = 0.025
    diverter_bounds = 0.5
    vacuum_id = 0
    chamber_id = 1

    # Global Coil parameters
    insulator_thickness = 0.001
    dielectric_thickness = 0.002

    # Interaction Coil parameters
    nr_interaction_coils = 8
    interaction_cable_thickness_r = 0.005
    interaction_cable_thickness_y = 0.004
    interaction_dielectric_id = 4
    interaction_switch_id = [5, 6, 7, 8, 9, 10, 11, 12]
    interaction_resistor_id = 13

    # Acceleration Coil parameters
    nr_acceleration_coils = 4
    accelerator_cable_thickness_r = 0.005
    accelerator_cable_thickness_y = 0.004
    accelerator_dielectric_id = 14  
    accelerator_switch_id = [15, 16, 17, 18]
    accelerator_resistor_id = 19

    # Formation Coil parameters
    nr_formation_coils = 4
    formation_cable_thickness_r = 0.005
    formation_cable_thickness_y = 0.004
    formation_dielectric_id = 20
    formation_switch_id = [21, 22, 23, 24]
    formation_resistor_id = 25

    # Inlet Diverter Coil parameters
    nr_diverter_inlet_coils = 2
    diverter_inlet_cable_thickness_r = 0.005
    diverter_inlet_cable_thickness_y = 0.004
    diverter_inlet_dielectric_id = 26
    diverter_inlet_switch_id = [27, 28]
    diverter_inlet_resistor_id = 29

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

    # Make the reactor operator
    reactor_operator = Reactor(
        chamber_wall_thickness=chamber_wall_thickness,
        interaction_radius=interaction_radius,
        interaction_bounds=interaction_bounds,
        acceleration_bounds=acceleration_bounds,
        formation_radius=formation_radius,
        formation_bounds=formation_bounds,
        diverter_inlet_radius=diverter_inlet_radius,
        diverter_inlet_bounds=diverter_inlet_bounds,
        diverter_radius=diverter_radius,
        diverter_bounds=diverter_bounds,
        insulator_thickness=insulator_thickness,
        dielectric_thickness=dielectric_thickness,
        vacuum_id=vacuum_id,
        chamber_id=chamber_id,
        conductor_id=2,
        insulator_id=3,
        nr_interaction_coils=nr_interaction_coils,
        interaction_cable_thickness_r=interaction_cable_thickness_r,
        interaction_cable_thickness_y=interaction_cable_thickness_y,
        interaction_resistor_id=interaction_resistor_id,
        interaction_switch_id=interaction_switch_id,
        interaction_dielectric_id=interaction_dielectric_id,
        nr_acceleration_coils=nr_acceleration_coils,
        accelerator_cable_thickness_r=accelerator_cable_thickness_r,
        accelerator_cable_thickness_y=accelerator_cable_thickness_y,
        accelerator_resistor_id=accelerator_resistor_id,
        accelerator_switch_id=accelerator_switch_id,
        accelerator_dielectric_id=accelerator_dielectric_id,
        nr_formation_coils=nr_formation_coils,
        formation_cable_thickness_r=formation_cable_thickness_r,
        formation_cable_thickness_y=formation_cable_thickness_y,
        formation_resistor_id=formation_resistor_id,
        formation_switch_id=formation_switch_id,
        formation_dielectric_id=formation_dielectric_id,
        nr_diverter_inlet_coils=nr_diverter_inlet_coils,
        diverter_inlet_cable_thickness_r=diverter_inlet_cable_thickness_r,
        diverter_inlet_cable_thickness_y=diverter_inlet_cable_thickness_y,
        diverter_inlet_resistor_id=diverter_inlet_resistor_id,
        diverter_inlet_switch_id=diverter_inlet_switch_id,
        diverter_inlet_dielectric_id=diverter_inlet_dielectric_id,
    )

    # Run the reactor
    id_field = reactor_operator(id_field)

    # Save the fields
    field_saver(
        {"id_field": id_field},
        "id_field.vtk",
    )
