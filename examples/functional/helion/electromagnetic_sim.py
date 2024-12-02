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

from geometry.reactor import Reactor

def capacitance_to_eps(
    capacitance: float,
    surface_area: float,
    thickness: float,
) -> float:
    return thickness * capacitance / surface_area

def switch_sigma_e(
    switch_sigma_e: float,
    switch_start_time: float,
    switch_time: float,
    switch_end_time: float,
) -> float:
    def _switch_sigma_e(t):
        if t < switch_start_time: # Before switch
            return 0.0
        elif t < switch_time + switch_start_time: # Ramp up
            alpha = (4.0 / switch_time)**2.0
            sigma_e = switch_sigma_e * np.exp(-alpha * (t - switch_time - switch_start_time)**2)
            return sigma_e
        elif t < switch_end_time: # Constant
            return switch_sigma_e
        elif t < switch_end_time + switch_time:
            alpha = (4.0 / switch_time)**2.0
            sigma_e = switch_sigma_e * (1.0 - np.exp(-alpha * (t - switch_end_time - switch_time)**2))
            return sigma_e
        else:
            return 0.0
    return _switch_sigma_e

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

class MaterialIDMappings(Operator):

    def __call__(
        self,
        eps_mapping: dict,
        mu_mapping: dict,
        sigma_e_mapping: dict,
        sigma_h_mapping: dict,
        time: float,
    ):

        # Make numpy arrays
        np_eps_mapping = np.zeros(len(list(eps_mapping.keys())), dtype=np.float32)
        np_mu_mapping = np.zeros(len(list(mu_mapping.keys())), dtype=np.float32)
        np_sigma_e_mapping = np.zeros(len(list(sigma_e_mapping.keys())), dtype=np.float32)
        np_sigma_h_mapping = np.zeros(len(list(sigma_h_mapping.keys())), dtype=np.float32)
        for key in eps_mapping.keys():
            if callable(eps_mapping[key]):
                np_eps_mapping[key] = eps_mapping[key](time)
            else:
                np_eps_mapping[key] = eps_mapping[key]
        for key in mu_mapping.keys():
            if callable(mu_mapping[key]):
                np_mu_mapping[key] = mu_mapping[key](time)
            else:
                np_mu_mapping[key] = mu_mapping[key]
        for key in sigma_e_mapping.keys():
            if callable(sigma_e_mapping[key]):
                np_sigma_e_mapping[key] = sigma_e_mapping[key](time)
            else:
                np_sigma_e_mapping[key] = sigma_e_mapping[key]
        for key in sigma_h_mapping.keys():
            if callable(sigma_h_mapping[key]):
                np_sigma_h_mapping[key] = sigma_h_mapping[key](time)
            else:
                np_sigma_h_mapping[key] = sigma_h_mapping[key]

        # Convert to warp fields
        wp_eps_mapping = wp.from_numpy(np_eps_mapping, wp.float32)
        wp_mu_mapping = wp.from_numpy(np_mu_mapping, wp.float32)
        wp_sigma_e_mapping = wp.from_numpy(np_sigma_e_mapping, wp.float32)
        wp_sigma_h_mapping = wp.from_numpy(np_sigma_h_mapping, wp.float32)

        return wp_eps_mapping, wp_mu_mapping, wp_sigma_e_mapping, wp_sigma_h_mapping



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
    dx = 0.002 # 1 mm
    origin = (-0.70, -0.10, -0.10) # meters
    spacing = (dx, dx, dx)
    shape = (int(1.4 / dx), int(0.20 / dx), int(0.20 / dx))
    nr_cells = shape[0] * shape[1] * shape[2]
    print(f"Number of million cells: {nr_cells / 1e6}")

    # IO parameters
    output_dir = "output_validation"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Constants
    vacuum_permittivity = 8.854187817e-12
    vacuum_permeability = 1.2566370614e-6
    vacuum_c = float(1.0 / np.sqrt(vacuum_permittivity * vacuum_permeability))
    copper_conductivity = 5.96e7
    insulator_permittivity = 3.45 * vacuum_permittivity

    # Material parameters
    eps_mapping = {}
    mu_mapping = {}
    sigma_e_mapping = {}
    sigma_h_mapping = {}
    initial_e_mapping = {}

    # Chamber parameters
    # Geometry
    chamber_wall_thickness = 0.0025
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
    # Material
    eps_mapping[vacuum_id] = vacuum_permittivity
    mu_mapping[vacuum_id] = vacuum_permeability
    sigma_e_mapping[vacuum_id] = 0
    sigma_h_mapping[vacuum_id] = 0
    eps_mapping[chamber_id] = vacuum_permittivity
    mu_mapping[chamber_id] = vacuum_permeability
    sigma_e_mapping[chamber_id] = 0
    sigma_h_mapping[chamber_id] = 0

    # Global Coil parameters
    # Geometry
    insulator_thickness = 0.002
    dielectric_thickness = 0.002
    insulator_id = 2
    conductor_id = 3
    # Material
    eps_mapping[insulator_id] = insulator_permittivity
    mu_mapping[insulator_id] = vacuum_permeability
    sigma_e_mapping[insulator_id] = 0
    sigma_h_mapping[insulator_id] = 0
    eps_mapping[conductor_id] = vacuum_permittivity
    mu_mapping[conductor_id] = vacuum_permeability
    sigma_e_mapping[conductor_id] = copper_conductivity
    sigma_h_mapping[conductor_id] = 0

    # Interaction Coil parameters
    # Geometry
    nr_interaction_coils = 8
    interaction_cable_thickness_r = 0.005
    interaction_cable_thickness_y = 0.004
    interaction_dielectric_id = 4
    interaction_switch_id = [5, 6, 7, 8, 9, 10, 11, 12]
    interaction_resistor_id = 13
    # Material
    interaction_voltage = 1000 # V
    interaction_capacitance = 1e-6 # F
    interation_switch_start_time = 1.0e-6 # s
    interaction_switch_time = 2e-7 # s
    interaction_switch_end_time = 1.5e-6 # s
    initial_e_mapping[interaction_dielectric_id] = interaction_voltage / dielectric_thickness
    eps_mapping[interaction_dielectric_id] = capacitance_to_eps(
        capacitance=interaction_capacitance,
        surface_area=interaction_cable_thickness_r * interaction_cable_thickness_y,
        thickness=dielectric_thickness,
    )
    mu_mapping[interaction_dielectric_id] = vacuum_permeability
    sigma_e_mapping[interaction_dielectric_id] = 0
    sigma_h_mapping[interaction_dielectric_id] = 0
    for switch_id in interaction_switch_id:
        eps_mapping[switch_id] = vacuum_permittivity
        mu_mapping[switch_id] = vacuum_permeability
        sigma_e_mapping[switch_id] = switch_sigma_e(
            switch_sigma_e=copper_conductivity,
            switch_start_time=interation_switch_start_time,
            switch_time=interaction_switch_time,
            switch_end_time=interaction_switch_end_time,
        )
        sigma_h_mapping[switch_id] = 0
    eps_mapping[interaction_resistor_id] = vacuum_permittivity
    mu_mapping[interaction_resistor_id] = vacuum_permeability
    sigma_e_mapping[interaction_resistor_id] = copper_conductivity
    sigma_h_mapping[interaction_resistor_id] = 0

    # Acceleration Coil parameters
    # Geometry
    nr_acceleration_coils = 4
    accelerator_cable_thickness_r = 0.005
    accelerator_cable_thickness_y = 0.004
    accelerator_dielectric_id = 14  
    accelerator_switch_id = [15, 16, 17, 18]
    accelerator_resistor_id = 19
    # Material
    accelerator_voltage = 1000 # V
    accelerator_capacitance = 1e-6 # F
    accelerator_switch_start_time = 5.0e-7 # s
    accelerator_switch_time = 2e-7 # s
    accelerator_switch_end_time = 1.0e-6 # s
    initial_e_mapping[accelerator_dielectric_id] = accelerator_voltage / dielectric_thickness
    eps_mapping[accelerator_dielectric_id] = capacitance_to_eps(
        capacitance=accelerator_capacitance,
        surface_area=accelerator_cable_thickness_r * accelerator_cable_thickness_y,
        thickness=dielectric_thickness,
    )
    mu_mapping[accelerator_dielectric_id] = vacuum_permeability
    sigma_e_mapping[accelerator_dielectric_id] = 0
    sigma_h_mapping[accelerator_dielectric_id] = 0
    for switch_id in accelerator_switch_id:
        eps_mapping[switch_id] = vacuum_permittivity
        mu_mapping[switch_id] = vacuum_permeability
        sigma_e_mapping[switch_id] = switch_sigma_e(
            switch_sigma_e=copper_conductivity,
            switch_start_time=accelerator_switch_start_time,
            switch_time=accelerator_switch_time,
            switch_end_time=accelerator_switch_end_time,
        )
        sigma_h_mapping[switch_id] = 0
    eps_mapping[accelerator_resistor_id] = vacuum_permittivity
    mu_mapping[accelerator_resistor_id] = vacuum_permeability
    sigma_e_mapping[accelerator_resistor_id] = copper_conductivity
    sigma_h_mapping[accelerator_resistor_id] = 0

    # Formation Coil parameters
    # Geometry
    nr_formation_coils = 4
    formation_cable_thickness_r = 0.005
    formation_cable_thickness_y = 0.004
    formation_dielectric_id = 20
    formation_switch_id = [21, 22, 23, 24]
    formation_resistor_id = 25
    # Material
    formation_voltage = 1000 # V
    formation_capacitance = 1e-6 # F
    formation_switch_start_time = 0.0 # s
    formation_switch_time = 2.0e-7 # s
    formation_switch_end_time = 5.0e-7 # s
    initial_e_mapping[formation_dielectric_id] = formation_voltage / dielectric_thickness
    eps_mapping[formation_dielectric_id] = capacitance_to_eps(
        capacitance=formation_capacitance,
        surface_area=formation_cable_thickness_r * formation_cable_thickness_y,
        thickness=dielectric_thickness,
    )
    mu_mapping[formation_dielectric_id] = vacuum_permeability
    sigma_e_mapping[formation_dielectric_id] = 0
    sigma_h_mapping[formation_dielectric_id] = 0
    for switch_id in formation_switch_id:
        eps_mapping[switch_id] = vacuum_permittivity
        mu_mapping[switch_id] = vacuum_permeability
        sigma_e_mapping[switch_id] = switch_sigma_e(
            switch_sigma_e=copper_conductivity,
            switch_start_time=formation_switch_start_time,
            switch_time=formation_switch_time,
            switch_end_time=formation_switch_end_time,
        )
        sigma_h_mapping[switch_id] = 0
    eps_mapping[formation_resistor_id] = vacuum_permittivity
    mu_mapping[formation_resistor_id] = vacuum_permeability
    sigma_e_mapping[formation_resistor_id] = copper_conductivity
    sigma_h_mapping[formation_resistor_id] = 0

    # Inlet Diverter Coil parameters
    # Geometry
    nr_diverter_inlet_coils = 2
    diverter_inlet_cable_thickness_r = 0.005
    diverter_inlet_cable_thickness_y = 0.004
    diverter_inlet_dielectric_id = 26
    diverter_inlet_switch_id = [27, 28]
    diverter_inlet_resistor_id = 29
    # Material
    diverter_inlet_voltage = 1000 # V
    diverter_inlet_capacitance = 1e-6 # F
    diverter_inlet_switch_start_time = 2.0e-7 # s
    diverter_inlet_switch_time = 5.0e-7 # s
    diverter_inlet_switch_end_time = 1.0e-6 # s
    initial_e_mapping[diverter_inlet_dielectric_id] = diverter_inlet_voltage / dielectric_thickness
    eps_mapping[diverter_inlet_dielectric_id] = capacitance_to_eps(
        capacitance=diverter_inlet_capacitance,
        surface_area=diverter_inlet_cable_thickness_r * diverter_inlet_cable_thickness_y,
        thickness=dielectric_thickness,
    )
    mu_mapping[diverter_inlet_dielectric_id] = vacuum_permeability
    sigma_e_mapping[diverter_inlet_dielectric_id] = 0
    sigma_h_mapping[diverter_inlet_dielectric_id] = 0
    for switch_id in diverter_inlet_switch_id:
        eps_mapping[switch_id] = vacuum_permittivity
        mu_mapping[switch_id] = vacuum_permeability
        sigma_e_mapping[switch_id] = switch_sigma_e(
            switch_sigma_e=copper_conductivity,
            switch_start_time=diverter_inlet_switch_start_time,
            switch_time=diverter_inlet_switch_time,
            switch_end_time=diverter_inlet_switch_end_time,
        )
        sigma_h_mapping[switch_id] = 0
    eps_mapping[diverter_inlet_resistor_id] = vacuum_permittivity
    mu_mapping[diverter_inlet_resistor_id] = vacuum_permeability
    sigma_e_mapping[diverter_inlet_resistor_id] = copper_conductivity
    sigma_h_mapping[diverter_inlet_resistor_id] = 0

    # PML parameters
    pml_width = 10

    # Use CFL condition to determine time step
    courant_number = 1.0 / np.sqrt(2.0) * 1.0
    ramp_dt = 0.1 * courant_number * (dx / vacuum_c)
    dt = 0.5 * courant_number * (dx / vacuum_c)

    # Time parameters
    ramp_time = 2e-9  # 10 nanoseconds
    relaxation_time = 1.0e-8  # 10 nanoseconds
    simulation_time = 2e-6  # 1 microsecond
    save_interval = 1e-8  # 10 nanoseconds
    save_frequency = int(save_interval / dt)

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
        conductor_id=conductor_id,
        insulator_id=insulator_id,
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

    # Make the electromagnetism operators
    e_field_update = YeeElectricFieldUpdate()
    h_field_update = YeeMagneticFieldUpdate()
    initialize_pml_layer = InitializePML()
    pml_e_field_update = PMLElectricFieldUpdate()
    pml_h_field_update = PMLMagneticFieldUpdate()
    pml_phi_e_update = PMLPhiEUpdate()
    pml_phi_h_update = PMLPhiHUpdate()
    ramp_electric_field = RampElectricField()
    material_id_mappings = MaterialIDMappings()

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

    # Run the reactor to initialize the id field
    id_field = reactor_operator(id_field)

    # Get material id mappings
    wp_eps_mapping, wp_mu_mapping, wp_sigma_e_mapping, wp_sigma_h_mapping = material_id_mappings(
        eps_mapping,
        mu_mapping,
        sigma_e_mapping,
        sigma_h_mapping,
        0.0,
    )

    # Run once to get the initial fields
    with wp.ScopedCapture() as capture:
        update_em_field(
            electric_field,
            magnetic_field,
            id_field,
            pml_layers,
            wp_eps_mapping,
            wp_mu_mapping,
            wp_sigma_e_mapping,
            wp_sigma_h_mapping,
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
        for e_id, e_strength in initial_e_mapping.items():
            electric_field = ramp_electric_field(
                electric_field,
                id_field,
                id_value=e_id,
                value=e_strength,
                direction=2,
            )

        # Call graph
        wp.capture_launch(capture.graph)

        # Check if nan
        if i % 100 == 0:
            if np.isnan(np.sum(electric_field.data.numpy())):
                print("Electric field is NaN")
                exit()

    # Save the fields
    field_saver(
        {
            "electric_field": electric_field,
            "magnetic_field": magnetic_field,
        },
        f"{output_dir}/ramped_em_field.vtk",
    )

    # Begin relaxation
    for i in tqdm(range(int(relaxation_time / dt))):

        # Ramp the electric field
        for e_id, e_strength in initial_e_mapping.items():
            electric_field = ramp_electric_field(
                electric_field,
                id_field,
                id_value=e_id,
                value=e_strength,
                direction=2,
            )

        # Call graph
        wp.capture_launch(capture.graph)

    # Save the fields
    field_saver(
        {
            "electric_field": electric_field,
            "magnetic_field": magnetic_field,
        },
        f"{output_dir}/relaxed_em_field.vtk",
    )

    # Begin simulation
    for i in tqdm(range(int(simulation_time / dt))):

        # Get material id mappings
        wp_eps_mapping, wp_mu_mapping, wp_sigma_e_mapping, wp_sigma_h_mapping = material_id_mappings(
            eps_mapping,
            mu_mapping,
            sigma_e_mapping,
            sigma_h_mapping,
            i * dt,
        )

        # Call the update function
        update_em_field(
            electric_field,
            magnetic_field,
            id_field,
            pml_layers,
            wp_eps_mapping,
            wp_mu_mapping,
            wp_sigma_e_mapping,
            wp_sigma_h_mapping,
            dt,
            pml_phi_e_update, 
            pml_phi_h_update,
            pml_e_field_update,
            pml_h_field_update,
            e_field_update,
            h_field_update,
        )

        # Save the fields
        if i % save_frequency == 0:

            # Save the fields
            print(f"Saving fields at time {i * dt}")
            field_saver(
                {
                    "id_field": id_field,
                    "electric_field": electric_field,
                    "magnetic_field": magnetic_field,
                },
                f"{output_dir}/simulation_em_field_{str(i).zfill(10)}.vtk",
            )
