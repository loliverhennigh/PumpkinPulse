# Helion fusion reactor simulation

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
from pumpkin_pulse.operator.hydro import (
    PrimitiveToConservative,
    ConservativeToPrimitive,
    GetTimeStep,
    EulerUpdate,
)
from pumpkin_pulse.operator.mhd import (
    AddEMSourceTerms,
    GetCurrentDensity,
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
from pumpkin_pulse.functional.indexing import periodic_indexing, periodic_setting
from pumpkin_pulse.functional.stencil import (
    p7_float32_stencil_type,
    p7_uint8_stencil_type,
    p4_float32_stencil_type,
    p4_uint8_stencil_type,
    faces_float32_type,
)
from pumpkin_pulse.operator.saver import FieldSaver

from geometry.circuit import CapacitorCircuit
from utils import (
    capacitance_to_eps,
    switch_sigma_e,
    RampElectricField,
    MaterialIDMappings,
    PlasmaInitializer,
    update_em_field,
    apply_boundary_conditions_p7,
    apply_boundary_conditions_faces,
)


if __name__ == "__main__":

    # Define simulation parameters
    dx = 0.0002 # 1 mm
    origin = (-0.70, -0.24, -0.12) # meters
    spacing = (dx, dx, dx)
    shape = (int(1.4 / dx), int(0.36 / dx), int(0.24 / dx))
    nr_cells = shape[0] * shape[1] * shape[2]
    print(f"Number of million cells: {nr_cells / 1e6}")

    # IO parameters
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Constants
    vacuum_permittivity = 8.854187817e-12
    vacuum_permeability = 1.2566370614e-6
    vacuum_c = float(1.0 / np.sqrt(vacuum_permittivity * vacuum_permeability))
    copper_conductivity = 5.96e7
    insulator_permittivity = 3.45 * vacuum_permittivity
    elementary_charge = 1.60217662e-19
    electron_mass = 9.10938356e-31
    proton_mass = 1.6726219e-27
    boltzmann_constant = 1.38064852e-23
    gamma = (5.0 / 3.0)

    # Plasma parameters
    initial_number_density = 1.0e19 # m^-3
    initial_temperature = 1.0e5 # K
    initial_pressure = initial_number_density * boltzmann_constant * initial_temperature

    # Material parameters
    eps_mapping = {}
    mu_mapping = {}
    sigma_e_mapping = {}
    sigma_h_mapping = {}
    initial_e_mapping = {}

    # Chamber parameters
    # Geometry
    chamber_wall_thickness = 0.020
    interaction_radius = 0.01
    interaction_bounds = 0.1
    acceleration_bounds = 0.3
    formation_radius = 0.025
    formation_bounds = 0.4
    diverter_inlet_radius = 0.01
    diverter_inlet_bounds = 0.45
    diverter_radius = 0.025
    diverter_bounds = 0.5
    background_id = 1
    vacuum_id = 0
    chamber_id = 2
    # Material
    eps_mapping[background_id] = vacuum_permittivity
    mu_mapping[background_id] = vacuum_permeability
    sigma_e_mapping[background_id] = 0
    sigma_h_mapping[background_id] = 0
    eps_mapping[vacuum_id] = vacuum_permittivity
    mu_mapping[vacuum_id] = vacuum_permeability
    sigma_e_mapping[vacuum_id] = 0
    sigma_h_mapping[vacuum_id] = 0
    eps_mapping[chamber_id] = 1.0 * vacuum_permittivity
    mu_mapping[chamber_id] = vacuum_permeability
    sigma_e_mapping[chamber_id] = 0
    sigma_h_mapping[chamber_id] = 0

    # Global Coil parameters
    # Geometry
    insulator_thickness = 0.002
    dielectric_thickness = 0.040
    insulator_id = 3
    conductor_id = 4
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
    interaction_cable_thickness_r = 0.020
    interaction_cable_thickness_y = 0.024
    interaction_dielectric_id = 5
    interaction_switch_id = [6, 7, 8, 9, 10, 11, 12, 13]
    interaction_resistor_id = 14
    # Material
    interaction_voltage = 1000 # V
    interaction_capacitance = 1.0 # F
    interation_switch_start_time = 1.0e-6 # s
    interaction_switch_time = 1e-7 # s
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
    accelerator_cable_thickness_r = 0.020
    accelerator_cable_thickness_y = 0.024
    accelerator_dielectric_id = 15  
    accelerator_switch_id = [16, 17, 18, 19]
    accelerator_resistor_id = 20
    # Material
    accelerator_voltage = 1000 # V
    accelerator_capacitance = 1.0 # F
    accelerator_switch_start_time = 5.0e-7 # s
    accelerator_switch_time = 1e-7 # s
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
    formation_cable_thickness_r = 0.020
    formation_cable_thickness_y = 0.024
    formation_dielectric_id = 21
    formation_switch_id = [22, 23, 24, 25]
    formation_resistor_id = 26
    # Material
    formation_voltage = 1000 # V
    formation_capacitance = 1.0 # F
    formation_switch_start_time = 0.0 # s
    formation_switch_time = 1.0e-7 # s
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
    diverter_inlet_cable_thickness_r = 0.012
    diverter_inlet_cable_thickness_y = 0.004
    diverter_inlet_dielectric_id = 27
    diverter_inlet_switch_id = [28, 29]
    diverter_inlet_resistor_id = 30
    # Material
    diverter_inlet_voltage = 1 # V
    diverter_inlet_capacitance = 1.0 # F
    diverter_inlet_switch_start_time = 2.0e-7 # s
    diverter_inlet_switch_time = 1.0e-7 # s
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

    # Make the id field
    id_field = constructor.create_field(
        dtype=wp.uint8,
        cardinality=1,
    )

    # Make EM fields
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

    # Make hydro fields
    hydro_offset = (
        int(((-diverter_bounds - chamber_wall_thickness) - origin[0]) / spacing[0]),
        int(((-formation_radius - chamber_wall_thickness) - origin[1]) / spacing[1]),
        int(((-formation_radius - chamber_wall_thickness) - origin[2]) / spacing[2]),
    )
    hydro_shape = (
        int((2 * (diverter_bounds + chamber_wall_thickness) / spacing[0]) + 0.5),
        int((2 * (formation_radius + chamber_wall_thickness) / spacing[1]) + 0.5),
        int((2 * (formation_radius + chamber_wall_thickness) / spacing[2]) + 0.5),
    )
    density_i = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        offset=hydro_offset,
        shape=hydro_shape,
    )
    velocity_i = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        offset=hydro_offset,
        shape=hydro_shape,
    )
    pressure_i = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        offset=hydro_offset,
        shape=hydro_shape,
    )
    density_e = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        offset=hydro_offset,
        shape=hydro_shape,
    )
    velocity_e = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        offset=hydro_offset,
        shape=hydro_shape,
    )
    pressure_e = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        offset=hydro_offset,
        shape=hydro_shape,
    )
    mass_i = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        offset=hydro_offset,
        shape=hydro_shape,
    )
    momentum_i = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        offset=hydro_offset,
        shape=hydro_shape,
    )
    energy_i = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        offset=hydro_offset,
        shape=hydro_shape,
    )
    mass_e = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        offset=hydro_offset,
        shape=hydro_shape,
    )
    momentum_e = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        offset=hydro_offset,
        shape=hydro_shape,
    )
    energy_e = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        offset=hydro_offset,
        shape=hydro_shape,
    )
    impressed_current = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        offset=hydro_offset,
        shape=hydro_shape,
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
        background_id=background_id,
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

    # Make the material id mappings operator
    material_id_mappings = MaterialIDMappings()

    # Make the electromagnetism operators
    e_field_update = YeeElectricFieldUpdate()
    h_field_update = YeeMagneticFieldUpdate()
    initialize_pml_layer = InitializePML()
    pml_e_field_update = PMLElectricFieldUpdate()
    pml_h_field_update = PMLMagneticFieldUpdate()
    pml_phi_e_update = PMLPhiEUpdate()
    pml_phi_h_update = PMLPhiHUpdate()
    ramp_electric_field = RampElectricField()

    # Make the hydro operators
    primitive_to_conservative = PrimitiveToConservative()
    conservative_to_primitive = ConservativeToPrimitive()
    get_time_step = GetTimeStep()
    euler_update = EulerUpdate(
        apply_boundary_conditions_p7=apply_boundary_conditions_p7,
        apply_boundary_conditions_faces=apply_boundary_conditions_faces,
    )
    add_em_source_terms = AddEMSourceTerms()
    get_current_density = GetCurrentDensity()
    plasma_initializer = PlasmaInitializer()

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
    field_saver(
        {
            "id_field": id_field,
        },
        f"{output_dir}/id_field.vtk",
    )

    # Get material id mappings
    wp_eps_mapping, wp_mu_mapping, wp_sigma_e_mapping, wp_sigma_h_mapping = material_id_mappings(
        eps_mapping,
        mu_mapping,
        sigma_e_mapping,
        sigma_h_mapping,
        0.0,
    )

    # Initialize the plasma
    density_i, velocity_i, pressure_i, density_e, velocity_e, pressure_e = plasma_initializer(
        density_i,
        velocity_i,
        pressure_i,
        density_e,
        velocity_e,
        pressure_e,
        id_field,
        proton_mass,
        electron_mass,
        initial_number_density,
        initial_pressure,
    )
    mass_i, momentum_i, energy_i = primitive_to_conservative(
        density_i,
        velocity_i,
        pressure_i,
        mass_i,
        momentum_i,
        energy_i,
        gamma
    )
    mass_e, momentum_e, energy_e = primitive_to_conservative(
        density_e,
        velocity_e,
        pressure_e,
        mass_e,
        momentum_e,
        energy_e,
        gamma
    )
    field_saver(
        {
            "density_i": density_i,
            "velocity_i": velocity_i,
            "pressure_i": pressure_i,
            "density_e": density_e,
            "velocity_e": velocity_e,
            "pressure_e": pressure_e,
            "mass_i": mass_i,
            "momentum_i": momentum_i,
            "energy_i": energy_i,
            "mass_e": mass_e,
            "momentum_e": momentum_e,
            "energy_e": energy_e,
        },
        f"{output_dir}/initial_plasma_hydro_fields.vtk",
    )

    # Run once to get the initial fields
    with wp.ScopedCapture() as capture:
        update_em_field(
            electric_field,
            magnetic_field,
            impressed_current,
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
    print("Beginning relaxation")
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
    print("Beginning relaxation")
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
    print("Beginning simulation")
    for i in tqdm(range(int(simulation_time / dt))):

        # Get primitive variables
        density_i, velocity_i, pressure_i = conservative_to_primitive(
            density_i,
            velocity_i,
            pressure_i,
            mass_i,
            momentum_i,
            energy_i,
            gamma
        )
        density_e, velocity_e, pressure_e = conservative_to_primitive(
            density_e,
            velocity_e,
            pressure_e,
            mass_e,
            momentum_e,
            energy_e,
            gamma
        )

        # Get impressed current
        impressed_current.data.zero_()
        impressed_current = get_current_density(
            density_i,
            velocity_i,
            pressure_i,
            impressed_current,
            id_field,
            proton_mass,
            elementary_charge,
        )
        impressed_current = get_current_density(
            density_e,
            velocity_e,
            pressure_e,
            impressed_current,
            id_field,
            electron_mass,
            -elementary_charge,
        )

        # Update Conserved Variables
        mass_i, momentum_i, energy_i = euler_update(
            density_i,
            velocity_i,
            pressure_i,
            mass_i,
            momentum_i,
            energy_i,
            id_field,
            gamma,
            dt,
        )
        mass_e, momentum_e, energy_e = euler_update(
            density_e,
            velocity_e,
            pressure_e,
            mass_e,
            momentum_e,
            energy_e,
            id_field,
            gamma,
            dt,
        )

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
            impressed_current,
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
        
        # Add EM Source Terms
        mass_i, momentum_i, energy_i = add_em_source_terms(
            density_i,
            velocity_i,
            pressure_i,
            electric_field,
            magnetic_field,
            mass_i,
            momentum_i,
            energy_i,
            id_field,
            proton_mass,
            elementary_charge,
            vacuum_permeability,
            dt,
        )
        mass_e, momentum_e, energy_e = add_em_source_terms(
            density_e,
            velocity_e,
            pressure_e,
            electric_field,
            magnetic_field,
            mass_e,
            momentum_e,
            energy_e,
            id_field,
            electron_mass,
            -elementary_charge,
            vacuum_permeability,
            dt,
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
            field_saver(
                {
                    "density_i": density_i,
                    "velocity_i": velocity_i,
                    "pressure_i": pressure_i,
                    "mass_i": mass_i,
                    "momentum_i": momentum_i,
                    "energy_i": energy_i,
                    "density_e": density_e,
                    "velocity_e": velocity_e,
                    "pressure_e": pressure_e,
                    "mass_e": mass_e,
                    "momentum_e": momentum_e,
                    "energy_e": energy_e,
                    "impressed_current": impressed_current,
                },
                f"{output_dir}/simulation_hydro_field_{str(i).zfill(10)}.vtk",
            )
