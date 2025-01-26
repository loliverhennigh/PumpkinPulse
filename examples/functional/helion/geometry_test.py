# Chamber

import os
import numpy as np
import warp as wp
import matplotlib.pyplot as plt
import dataclasses
import itertools
from tqdm import tqdm

wp.init()
#wp.clear_kernel_cache()

from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.data import Fielduint8, Fieldfloat32
from pumpkin_pulse.operator import Operator
from pumpkin_pulse.operator.voxelize import (
    Tube,
)
from pumpkin_pulse.operator.saver import FieldSaver

from geometry.chamber import HelionChamber
from geometry.circuit import CapacitorCircuit
#from geometry.circuit import test


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
    interaction_coil_radius = interaction_radius + chamber_wall_thickness
    interaction_cable_thickness_r = 0.005
    interaction_cable_thickness_y = 0.004
    interaction_positions = np.linspace(-interaction_bounds, interaction_bounds, nr_interaction_coils)
    interaction_conductor_id = 2
    interaction_insulator_id = 3
    interaction_dielectric_id = 4
    interaction_switch_id = 5
    interaction_resistor_id = 6

    # Acceleration Coil parameters
    nr_acceleration_coils = 4
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
    accelerator_cable_thickness_r = 0.005
    accelerator_cable_thickness_y = 0.004
    accelerator_conductor_id = 7
    accelerator_insulator_id = 8
    accelerator_dielectric_id = 9
    accelerator_switch_id = 10
    accelerator_resistor_id = 11

    # Formation Coil parameters
    nr_formation_coils = 4
    formation_coil_radius = formation_radius + chamber_wall_thickness
    formation_cable_thickness_r = 0.005
    formation_cable_thickness_y = 0.004
    formation_positions = np.linspace(
        acceleration_bounds,
        formation_bounds,
        nr_formation_coils
    )
    formation_conductor_id = 12
    formation_insulator_id = 13
    formation_dielectric_id = 14
    formation_switch_id = 15
    formation_resistor_id = 16

    # Inlet Diverter Coil parameters
    nr_diverter_inlet_coils = 2
    diverter_inlet_coil_radius = diverter_inlet_radius + chamber_wall_thickness
    diverter_inlet_cable_thickness_r = 0.005
    diverter_inlet_cable_thickness_y = 0.004
    diverter_inlet_positions = np.linspace(
        formation_bounds + chamber_wall_thickness + 0.5 * diverter_inlet_cable_thickness_y,
        diverter_inlet_bounds - chamber_wall_thickness - 0.5 * diverter_inlet_cable_thickness_y,
        nr_diverter_inlet_coils
    )
    diverter_inlet_conductor_id = 17
    diverter_inlet_insulator_id = 18
    diverter_inlet_dielectric_id = 19
    diverter_inlet_switch_id = 20
    diverter_inlet_resistor_id = 21

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

    # Make the Helion chamber operator
    helion_chamber_operator = HelionChamber(
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

    # Make interaction coils operators
    interaction_coils_operators = []
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
            conductor_id=interaction_conductor_id,
            insulator_id=interaction_insulator_id,
            dielectric_id=interaction_dielectric_id,
            switch_id=interaction_switch_id,
            resistor_id=interaction_resistor_id,
        )
        interaction_coils_operators.append(interaction_coil_operator)

    # Make acceleration coils operators
    lx_acceleration_coils_operators = []
    rx_acceleration_coils_operators = []
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
            conductor_id=accelerator_conductor_id,
            insulator_id=accelerator_insulator_id,
            dielectric_id=accelerator_dielectric_id,
            switch_id=accelerator_switch_id,
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
            conductor_id=accelerator_conductor_id,
            insulator_id=accelerator_insulator_id,
            dielectric_id=accelerator_dielectric_id,
            switch_id=accelerator_switch_id,
            resistor_id=accelerator_resistor_id,
        )
        lx_acceleration_coils_operators.append(lx_acceleration_coil_operator)
        rx_acceleration_coils_operators.append(rx_acceleration_coil_operator)

    # Make formation coils operators
    lx_formation_coils_operators = []
    rx_formation_coils_operators = []
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
            conductor_id=formation_conductor_id,
            insulator_id=formation_insulator_id,
            dielectric_id=formation_dielectric_id,
            switch_id=formation_switch_id,
            resistor_id=formation_resistor_id,
        )
        lx_formation_coils_operators.append(lx_formation_coil_operator)
        rx_formation_coil_operator = CapacitorCircuit(
            coil_radius=formation_coil_radius,
            cable_thickness_r=formation_cable_thickness_r,
            cable_thickness_y=formation_cable_thickness_y,
            insulator_thickness=insulator_thickness,
            dielectric_thickness=dielectric_thickness,
            center=(-float(formation_positions[i]), 0.0, 0.0),
            centeraxis=(0.0, 0.0, 1.0),
            angle=np.pi / 2,
            conductor_id=formation_conductor_id,
            insulator_id=formation_insulator_id,
            dielectric_id=formation_dielectric_id,
            switch_id=formation_switch_id,
            resistor_id=formation_resistor_id,
        )
        rx_formation_coils_operators.append(rx_formation_coil_operator)

    # Make diverter inlet coils operators
    lx_diverter_inlet_coils_operators = []
    rx_diverter_inlet_coils_operators = []
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
            conductor_id=diverter_inlet_conductor_id,
            insulator_id=diverter_inlet_insulator_id,
            dielectric_id=diverter_inlet_dielectric_id,
            switch_id=diverter_inlet_switch_id,
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
            conductor_id=diverter_inlet_conductor_id,
            insulator_id=diverter_inlet_insulator_id,
            dielectric_id=diverter_inlet_dielectric_id,
            switch_id=diverter_inlet_switch_id,
            resistor_id=diverter_inlet_resistor_id,
        )
        lx_diverter_inlet_coils_operators.append(lx_diverter_inlet_coil_operator)
        rx_diverter_inlet_coils_operators.append(rx_diverter_inlet_coil_operator)

 
    # Run the geometry operators
    id_field = helion_chamber_operator(id_field)
    for interaction_coil_operator in interaction_coils_operators:
        id_field = interaction_coil_operator(id_field)
    for lx_acceleration_coil_operator in lx_acceleration_coils_operators:
        id_field = lx_acceleration_coil_operator(id_field)
    for rx_acceleration_coil_operator in rx_acceleration_coils_operators:
        id_field = rx_acceleration_coil_operator(id_field)
    for lx_formation_coil_operator in lx_formation_coils_operators:
        id_field = lx_formation_coil_operator(id_field)
    for rx_formation_coil_operator in rx_formation_coils_operators:
        id_field = rx_formation_coil_operator(id_field)
    for lx_diverter_inlet_coil_operator in lx_diverter_inlet_coils_operators:
        id_field = lx_diverter_inlet_coil_operator(id_field)
    for rx_diverter_inlet_coil_operator in rx_diverter_inlet_coils_operators:
        id_field = rx_diverter_inlet_coil_operator(id_field)

    # Save the fields
    field_saver(
        {"id_field": id_field},
        "id_field.vtk",
    )
