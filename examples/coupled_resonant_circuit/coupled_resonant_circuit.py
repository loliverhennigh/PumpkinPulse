# Coupled resonant circuit simulation using Pumpkin Pulse

import os
import numpy as np
import warp as wp
import matplotlib.pyplot as plt
from tqdm import tqdm

wp.init()

from pumpkin_pulse.ds import AMRGrid
from pumpkin_pulse.geometry.circuit.circular_rlc import CircularRLC
from pumpkin_pulse.material.material import (
    COPPER,
)
from pumpkin_pulse.subroutine.amr_grid.voxelize import (
    VoxelizerSubroutine,
)
from pumpkin_pulse.subroutine.amr_grid.electromagnetism import (
    PMLInitializerSubroutine,
)

if __name__ == "__main__":

    # Define simulation parameters
    dx = 0.002
    origin = (0.0, 0.0, 0.0) # meters
    spacing = (dx, dx, dx)
    shape = (int(1.0 / dx), int(1.0 / dx), int(1.0 / dx))
    block_shape = (250, 250, 250)
    ghost_cell_thickness = 16

    # Define the coil parameters
    coil_radius = 0.2
    cable_thickness_r = 0.05
    cable_thickness_y = 0.05
    insulator_thickness = 0.05
    dielectric_thickness = 0.1
    center_1 = (0.5, 0.25, 0.5)
    center_2 = (0.5, 0.75, 0.5)

    # Define the circuit parameters 
    voltage = 100.0
    capacitance = 1e-6
    resistance = 1e-6
    peak_switch_electrical_conductivity = COPPER.electrical_conductivity
    switch_start_time = 0.0
    switch_end_time = 1e6 # Always on
    switch_time = 1e-6 # 1 microsecond

    # Electromagnetic parameters
    pml_thickness = 8
    courant_number = 0.5

    # Make the circular RLC circuits
    circuit_1 = CircularRLC(
        coil_radius=coil_radius,
        cable_thickness_r=cable_thickness_r,
        cable_thickness_y=cable_thickness_y,
        insulator_thickness=insulator_thickness,
        dielectric_thickness=dielectric_thickness,
        voltage=voltage,
        capacitance=capacitance,
        resistance=resistance,
        peak_switch_electrical_conductivity=peak_switch_electrical_conductivity,
        switch_start_time=switch_start_time,
        switch_end_time=switch_end_time,
        switch_time=switch_time,
        center_0=center_1,
        name="circuit_rlc_1",
    )
    circuit_2 = CircularRLC(
        coil_radius=coil_radius,
        cable_thickness_r=cable_thickness_r,
        cable_thickness_y=cable_thickness_y,
        insulator_thickness=insulator_thickness,
        dielectric_thickness=dielectric_thickness,
        voltage=0.0,
        capacitance=capacitance,
        resistance=resistance,
        peak_switch_electrical_conductivity=peak_switch_electrical_conductivity,
        switch_start_time=-switch_time, # Switch already on
        switch_end_time=switch_end_time,
        switch_time=switch_time,
        center_0=center_2,
        name="circuit_rlc_2",
    )

    # Make geometry
    geometry = circuit_1 + circuit_2

    # Make amr grid
    amr_grid = AMRGrid(
        shape=shape,
        block_shape=block_shape,
        origin=origin,
        spacing=spacing,
        ghost_cell_thickness=ghost_cell_thickness,
        comm=None,
    )

    # Initialize boxes
    amr_grid.initialize_boxes(
        "id_field",
        dtype=wp.int16,
        cardinality=1,
    )
    amr_grid.initialize_boxes(
        "electric_field",
        dtype=wp.float32,
        cardinality=3,
    )
    amr_grid.initialize_boxes(
        "magnetic_field",
        dtype=wp.float32,
        cardinality=3,
    )
    amr_grid.initialize_boxes(
        "temperature_field",
        dtype=wp.float32,
        cardinality=1,
    )
    pml_layer_names = []
    for dim in ["x", "y", "z"]:
        for side in ["low", "high"]:

            # Get offset and extent
            offset = np.array(
                [
                    0 if (side == "low" or dim != "x") else shape[0] - pml_thickness,
                    0 if (side == "low" or dim != "y") else shape[1] - pml_thickness,
                    0 if (side == "low" or dim != "z") else shape[2] - pml_thickness,
                ]
            )
            extent = np.array(
                [
                    pml_thickness if dim == "x" else shape[0],
                    pml_thickness if dim == "y" else shape[1],
                    pml_thickness if dim == "z" else shape[2],
                ]
            )

            # Set the PML
            amr_grid.initialize_boxes(
                f"pml_{dim}_{side}",
                dtype=wp.float32,
                cardinality=36,
                offset=offset,
                extent=extent,
            )

            # Append to the list
            pos_neg = -1.0 if side == "low" else 1.0
            direction = wp.vec3(
                pos_neg if dim == "x" else 0.0,
                pos_neg if dim == "y" else 0.0,
                pos_neg if dim == "z" else 0.0,
            )
            pml_layer_names.append(
                (
                    f"pml_{dim}_{side}",
                    direction,
                    pml_thickness,
                )
            )

    # Allocate data
    amr_grid.allocate()

    # Get property mappings
    property_mappings = geometry.get_property_mappings()

    # Make operators
    voxelizer = geometry.get_voxelizer()
    property_mapping_updator = geometry.get_property_mapping_updater()

    # Set the property mappings for t=0
    property_mappings = property_mapping_updator(property_mappings, 0.0)

    # Make subroutines
    voxelizer_subroutine = VoxelizerSubroutine(
        voxelizer=voxelizer,
    )
    pml_initializer_subroutine = PMLInitializerSubroutine()

    # Initialize the id field
    voxelizer_subroutine(
        amr_grid,
        "id_field",
    )

    # Initialize the PML
    pml_initializer_subroutine(
        amr_grid,
        pml_layer_names,
        courant_number=courant_number,
        k=1.0,
        a=1.0e-8,
    )

    # Save the initial state
    #amr_grid.save_vtm("initial_state.vtm")
