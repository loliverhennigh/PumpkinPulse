# Electromagnet geometry test

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
from pumpkin_pulse.operator.geometry.circuit import (
    Capacitor,
    Coil,
    Element,
    Cable,
)
from pumpkin_pulse.operator.electromagnetism import (
    YeeElectricFieldUpdate,
    YeeMagneticFieldUpdate,
    InitializePML,
    PMLElectricFieldUpdate,
    PMLMagneticFieldUpdate,
    PMLPhiEUpdate,
    PMLPhiHUpdate,
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
    )

    # Make the resistor
    resistor = Element(
        cable_radius=cable_radius,
        insulator_thickness=insulator_thickness,
        element_length=3.0 * cable_radius,
        element_id=resistor_id,
        insulator_id=insulator_id,
        center=(
            center[0]
            - (
                coil_radius
                + 2.0 * (nr_turns_r + 2.0) * (cable_radius + insulator_thickness)
            ),
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
            center[0]
            - (
                coil_radius
                + 2.0 * (nr_turns_r + 2.0) * (cable_radius + insulator_thickness)
            ),
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
            center[0]
            - (
                coil_radius
                + 2.0 * (nr_turns_r + 2.0) * (cable_radius + insulator_thickness)
            ),
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
        scale=1.5 * (coil_radius + cable_radius + insulator_thickness),
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
        scale=1.5 * (coil_radius + cable_radius + insulator_thickness),
    )

    # Add the operators to the list
    operators.append(coil)
    operators.append(resistor)
    operators.append(switch)
    operators.append(capacitor)
    operators.append(cable_0)
    operators.append(cable_1)

    return operators

class RampElectricField(Operator):

    @wp.kernel
    def _ramp_electric_field(
        electric_field: Fieldfloat32,
        id_field: Fielduint8,
        id_value: wp.uint8,
        value: wp.float32,
        direction: wp.int32,
    ):
        # Get index
        i, j, k = wp.tid()

        # Get id value
        local_id = id_field.data[0, i, j, k]

        # Check if id value is equal to the id value
        if id_value == local_id:

            # Set the electric field including edges
            if direction == 0:
                electric_field.data[0, i, j, k] = value
                electric_field.data[0, i, j, k + 1] = value
                electric_field.data[0, i, j + 1, k] = value
                electric_field.data[0, i, j + 1, k + 1] = value
            elif direction == 1:
                electric_field.data[1, i, j, k] = value
                electric_field.data[1, i + 1, j, k] = value
                electric_field.data[1, i, j, k + 1] = value
                electric_field.data[1, i + 1, j, k + 1] = value
            elif direction == 2:
                electric_field.data[2, i, j, k] = value
                electric_field.data[2, i, j + 1, k] = value
                electric_field.data[2, i + 1, j, k] = value
                electric_field.data[2, i + 1, j + 1, k] = value

    def __call__(
        self,
        electric_field: Fieldfloat32,
        id_field: Fielduint8,
        id_value: wp.uint8,
        value: wp.float32,
        direction: wp.int32,
    ):
        # Launch kernel
        wp.launch(
            self._ramp_electric_field,
            inputs=[
                electric_field,
                id_field,
                id_value,
                value,
                direction,
            ],
            dim=[s - 1 for s in electric_field.shape],
        )

        return electric_field

def update_em_field(
    electric_field,
    magnetic_field,
    id_field,
    pml_layers,
    eps_mapping,
    mu_mapping,
    sigma_e_mapping,
    sigma_h_mapping,
    dt,
    pml_phi_e_update, 
    pml_phi_h_update,
    pml_e_field_update,
    pml_h_field_update,
    e_field_update,
    h_field_update,
):

    # Update the PML phi_e fields
    for pml_layer in pml_layers:
        pml_layer = pml_phi_e_update(
            magnetic_field,
            pml_layer,
        )

    # Update the electric field
    electric_field = e_field_update(
        electric_field,
        magnetic_field,
        None,
        id_field,
        eps_mapping,
        sigma_e_mapping,
        dt
    )

    # Update the electric field with PML
    for pml_layer in pml_layers:
        electric_field = pml_e_field_update(
            electric_field,
            pml_layer,
            id_field,
            eps_mapping,
            dt,
        )

    # Update the PML phi_h fields
    for pml_layer in pml_layers:
        pml_layer = pml_phi_h_update(
            electric_field,
            pml_layer,
        )

    # Update the magnetic field
    magnetic_field = h_field_update(
        electric_field,
        magnetic_field,
        id_field,
        mu_mapping,
        sigma_h_mapping,
        dt
    )

    # Update the magnetic field with PML
    for pml_layer in pml_layers:
        magnetic_field = pml_h_field_update(
            magnetic_field,
            pml_layer,
            id_field,
            mu_mapping,
            dt,
        )

if __name__ == "__main__":
    # Define simulation parameters
    dx = 0.01
    origin = (-1.0, -1.0, -1.0)
    spacing = (dx, dx, dx)
    shape = (int(3.0 / dx), int(3.0 / dx), int(3.0 / dx))
    nr_cells = shape[0] * shape[1] * shape[2]
    print(f"Number of cells: {nr_cells}")

    # IO parameters
    output_dir = "output_validation"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Coil Capacitor circuit parameters
    coil_radius = 0.4
    cable_radius = 0.02
    insulator_thickness = 0.02
    nr_turns_z = 0
    nr_turns_r = 1
    center = (0.6, 0.5, 0.5)
    resistance = 1.0  # Ohm
    voltage = 1.0e3  # Volt
    capacitance = 1.0e-6  # Farad

    # Electromagnet parameters
    # Vacuum parameters
    vacuum_eps = 8.854187817e-12
    vacuum_mu = 4.0 * np.pi * 1.0e-7
    vacuum_c = 1.0 / np.sqrt(vacuum_eps * vacuum_mu)
    vacuum_sigma_e = 0.0
    vacuum_sigma_h = 0.0

    # Insulator parameters
    insulator_eps = 3.0 * vacuum_eps
    insulator_mu = vacuum_mu
    insulator_sigma_e = 0.0
    insulator_sigma_h = 0.0

    # Conductor parameters (copper)
    conductor_eps = vacuum_eps
    conductor_mu = vacuum_mu
    conductor_sigma_e = 5.8e7  # S/m
    conductor_sigma_h = 0.0

    # Resistor parameters
    resistor_length = 3.0 * cable_radius
    resistor_eps = vacuum_eps
    resistor_mu = vacuum_mu
    #resistor_sigma_e = resistor_length / (resistance * np.pi * cable_radius**2)
    #resistor_sigma_e = conductor_sigma_e
    resistor_sigma_e = 0.0
    print(f"Conductor conductivity: {conductor_sigma_e}")
    print(f"Resistor conductivity: {resistor_sigma_e}")
    resistor_sigma_h = 0.0

    # Switch parameters
    switch_eps = vacuum_eps
    switch_mu = vacuum_mu
    switch_sigma_e = 5.8e7  # S/m
    switch_sigma_h = 0.0

    # Dielectric parameters
    dielectric_thickness = cable_radius
    dielectric_surface_area = (5.0 * cable_radius) ** 2
    dielectric_e_strength = voltage / dielectric_thickness
    dielectric_eps = (capacitance * dielectric_thickness) / dielectric_surface_area
    dielectric_mu = vacuum_mu
    dielectric_sigma_e = 0.0
    dielectric_sigma_h = 0.0

    # PML parameters
    pml_width = 10

    # Use CFL condition to determine time step
    courant_number = 1.0 / np.sqrt(2.0) * 1.0
    ramp_dt = 0.1 * courant_number * (dx / vacuum_c)
    dt = 0.5 * courant_number * (dx / vacuum_c)

    # Time parameters
    ramp_time = 1e-8  # 100 nanoseconds
    relaxation_time = 1.0e-8  # 100 nanoseconds
    switch_time = 5e-7  # 100 nanoseconds
    simulation_time = 1e-6  # 1 microsecond

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
    electric_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
    )
    magnetic_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
    )
    pml_layer_lower_x = constructor.create_field(
        dtype=wp.float32,
        cardinality=36,
        offset=(0, 0, 0),
        shape=(pml_width, shape[1], shape[2]),
    )
    pml_layer_upper_x = constructor.create_field(
        dtype=wp.float32,
        cardinality=36,
        offset=(shape[0] - pml_width, 0, 0),
        shape=(pml_width, shape[1], shape[2]),
    )
    pml_layer_lower_y = constructor.create_field(
        dtype=wp.float32,
        cardinality=36,
        offset=(0, 0, 0),
        shape=(shape[0], pml_width, shape[2]),
    )
    pml_layer_upper_y = constructor.create_field(
        dtype=wp.float32,
        cardinality=36,
        offset=(0, shape[1] - pml_width, 0),
        shape=(shape[0], pml_width, shape[2]),
    )
    pml_layer_lower_z = constructor.create_field(
        dtype=wp.float32,
        cardinality=36,
        offset=(0, 0, 0),
        shape=(shape[0], shape[1], pml_width),
    )
    pml_layer_upper_z = constructor.create_field(
        dtype=wp.float32,
        cardinality=36,
        offset=(0, 0, shape[2] - pml_width),
        shape=(shape[0], shape[1], pml_width),
    )
    pml_layers = [
        pml_layer_lower_x,
        pml_layer_upper_x,
        pml_layer_lower_y,
        pml_layer_upper_y,
        pml_layer_lower_z,
        pml_layer_upper_z,
    ]

    # Make material property mappings
    eps_mapping = wp.from_numpy(
        np.array(
            [
                vacuum_eps,
                conductor_eps,
                insulator_eps,
                resistor_eps,
                dielectric_eps,
                switch_eps,
            ],
            dtype=np.float32,
        ),
        dtype=wp.float32,
    )
    mu_mapping = wp.from_numpy(
        np.array(
            [
                vacuum_mu,
                conductor_mu,
                insulator_mu,
                resistor_mu,
                dielectric_mu,
                switch_mu,
            ],
            dtype=np.float32,
        ),
        dtype=wp.float32,
    )
    sigma_e_mapping = wp.from_numpy(
        np.array(
            [
                vacuum_sigma_e,
                conductor_sigma_e,
                insulator_sigma_e,
                resistor_sigma_e,
                dielectric_sigma_e,
                0.0, # start with no conductivity in the switch
            ],
            dtype=np.float32,
        ),
        dtype=wp.float32,
    )
    sigma_h_mapping = wp.from_numpy(
        np.array(
            [
                vacuum_sigma_h,
                conductor_sigma_h,
                insulator_sigma_h,
                resistor_sigma_h,
                dielectric_sigma_h,
                switch_sigma_h,
            ],
            dtype=np.float32,
        ),
        dtype=wp.float32,
    )

    # Make the geometry operators
    geometry_operators = make_circuit(
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

    # Make the electromagnetism operators
    e_field_update = YeeElectricFieldUpdate()
    h_field_update = YeeMagneticFieldUpdate()
    initialize_pml_layer = InitializePML()
    pml_e_field_update = PMLElectricFieldUpdate()
    pml_h_field_update = PMLMagneticFieldUpdate()
    pml_phi_e_update = PMLPhiEUpdate()
    pml_phi_h_update = PMLPhiHUpdate()
    ramp_electric_field = RampElectricField()

    # Initialize the PML layers
    pml_layer_lower_x = initialize_pml_layer(
        pml_layer_lower_x,
        direction=wp.vec3f(1.0, 0.0, 0.0),
        thickness=pml_width,
        courant_number=courant_number,
    )
    pml_layer_upper_x = initialize_pml_layer(
        pml_layer_upper_x,
        direction=wp.vec3f(-1.0, 0.0, 0.0),
        thickness=pml_width,
        courant_number=courant_number,
    )
    pml_layer_lower_y = initialize_pml_layer(
        pml_layer_lower_y,
        direction=wp.vec3f(0.0, 1.0, 0.0),
        thickness=pml_width,
        courant_number=courant_number,
    )
    pml_layer_upper_y = initialize_pml_layer(
        pml_layer_upper_y,
        direction=wp.vec3f(0.0, -1.0, 0.0),
        thickness=pml_width,
        courant_number=courant_number,
    )
    pml_layer_lower_z = initialize_pml_layer(
        pml_layer_lower_z,
        direction=wp.vec3f(0.0, 0.0, 1.0),
        thickness=pml_width,
        courant_number=courant_number,
    )
    pml_layer_upper_z = initialize_pml_layer(
        pml_layer_upper_z,
        direction=wp.vec3f(0.0, 0.0, -1.0),
        thickness=pml_width,
        courant_number=courant_number
    )

    # Initialize id field
    for operator in geometry_operators:
        id_field = operator(id_field)

    # Save the fields
    field_saver(
        {"id_field": id_field},
        f"{output_dir}/id_field.vtk",
    )

    # Run once to get the initial fields
    with wp.ScopedCapture() as capture:
        update_em_field(
            electric_field,
            magnetic_field,
            id_field,
            pml_layers,
            eps_mapping,
            mu_mapping,
            sigma_e_mapping,
            sigma_h_mapping,
            dt,
            pml_phi_e_update, 
            pml_phi_h_update,
            pml_e_field_update,
            pml_h_field_update,
            e_field_update,
            h_field_update,
        )

    # Begin field rampup
    for i in tqdm(range(int(ramp_time / ramp_dt))):

        # Ramp the electric field
        alpha = (4.0 / ramp_time)**2.0
        e_strength = dielectric_e_strength * np.exp(-alpha * (i * ramp_dt - ramp_time)**2.0)
        electric_field = ramp_electric_field(
            electric_field,
            id_field,
            id_value=4,
            value=e_strength,
            direction=1,
        )

        # Call graph
        wp.capture_launch(capture.graph)

    # Save the fields
    field_saver(
        {
            "electric_field": electric_field,
            "magnetic_field": magnetic_field,
        },
        f"{output_dir}/ramped_em_field.vtk",
    )

    # Begin relaxation
    prev_electric_field = electric_field.data.numpy()
    prev_magnetic_field = magnetic_field.data.numpy()
    for i in tqdm(range(int(relaxation_time / dt))):

        # Ramp the electric field
        electric_field = ramp_electric_field(
            electric_field,
            id_field,
            id_value=4,
            value=dielectric_e_strength,
            direction=1,
        )

        # Call graph
        wp.capture_launch(capture.graph)

    # Begin simulation
    for i in tqdm(range(int(simulation_time / dt))):

        # If the switch time has been reached, add conductivity to the switch
        if i * dt < switch_time:

            alpha = (4.0 / switch_time)**2.0
            new_sigma_e = switch_sigma_e * np.exp(-alpha * (i * dt - switch_time)**2)
            sigma_e_mapping = wp.from_numpy(
                np.array(
                    [
                        vacuum_sigma_e,
                        conductor_sigma_e,
                        insulator_sigma_e,
                        new_sigma_e,
                        dielectric_sigma_e,
                        new_sigma_e,
                    ],
                    dtype=np.float32,
                ),
                dtype=wp.float32,
            )
            print(f"New sigma_e: {new_sigma_e}")

        # Call graph
        wp.capture_launch(capture.graph)

        # Compute the residual
        if i % 1000 == 0:

            # Save the fields
            field_saver(
                {
                    "id_field": id_field,
                    "electric_field": electric_field,
                    "magnetic_field": magnetic_field,
                },
                f"{output_dir}/simulation_em_field_{str(i).zfill(10)}.vtk",
            )




