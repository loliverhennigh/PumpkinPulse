# Dense fusion reactor simulation

import os
import numpy as np
import warp as wp
import matplotlib.pyplot as plt
import dataclasses
import itertools
from tqdm import tqdm
from typing import List
from anytree.search import findall
from build123d import Part
import trimesh
import tempfile
import subprocess
import os

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
from pumpkin_pulse.operator.mesh import MeshToIdField

from pumpkin_pulse.geometry.reactor.reactor import LLPReactor

from ion_euler import IonEulerUpdate
from utils import (
    capacitance_to_eps,
    switch_sigma_e,
    RampElectricField,
    MaterialIDMappings,
    PlasmaInitializer,
    update_em_field,
    apply_boundary_conditions_p7,
    apply_boundary_conditions_faces,
    FillIdField,
)

field_saver = FieldSaver()

#def save_to_temp_and_copy(field_data, remote_path):
#    field_saver(field_data, remote_path)


# Define a helper function to save fields via a temp file and then copy them
def save_to_temp_and_copy(field_data, remote_path):
    # remote_path should be something like "/path/on/remote/host/filename.vtk"
    # Make a local temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".vtk") as tmp_file:
        temp_filename = tmp_file.name

    # Save the data locally using field_saver
    field_saver(field_data, temp_filename)

    # Copy the file to the remote location using scp
    scp_command = ["scp", temp_filename, f"oliver@pumpkinlatte:{remote_path}"]
    subprocess.run(scp_command, check=True)

    # Remove the temporary file
    os.remove(temp_filename)


if __name__ == "__main__":

    # Define simulation parameters
    dx = 0.00035 # 1 mm
    #dx = 0.00070 # 1 mm
    origin = (-0.035, -0.035, -0.055) # meters
    spacing = (dx, dx, dx)
    shape = (int(0.070 / dx), int(0.07 / dx), int(0.12 / dx))
    nr_cells = shape[0] * shape[1] * shape[2]
    print(f"Number of million cells: {nr_cells / 1e6}")

    # IO parameters
    output_dir = "/home/oliver/dpf_output/"
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
    initial_number_density = 3.0e20 # m^-3
    initial_temperature = 1.0e5 # K
    initial_pressure = initial_number_density * boltzmann_constant * initial_temperature

    # Material parameters
    eps_mapping = {}
    mu_mapping = {}
    sigma_e_mapping = {}
    sigma_h_mapping = {}
    initial_e_mapping = {}
    eps_mapping[0] = vacuum_permittivity
    mu_mapping[0] = vacuum_permeability
    sigma_e_mapping[0] = 0.0
    sigma_h_mapping[0] = 0.0
    eps_mapping[1] = insulator_permittivity
    mu_mapping[1] = vacuum_permeability
    sigma_e_mapping[1] = 0.0
    sigma_h_mapping[1] = 0.0

    # Reactor parameters
    voltage = 1000.0
    reactor = LLPReactor(
        resistor_conductivity=switch_sigma_e(
            switch_sigma_e=copper_conductivity,
            switch_start_time=0.0,
            switch_time=1.0e-7,
            switch_end_time=1.0,
        ),
        voltage=voltage,
        capacitance=1.0/voltage, # 1 Coulomb
    )

    # PML parameters
    pml_width = 10

    # Use CFL condition to determine time step
    courant_number = 1.0 / np.sqrt(2.0) * 1.0
    ramp_dt = 0.1 * courant_number * (dx / vacuum_c)
    dt = 0.5 * courant_number * (dx / vacuum_c)

    # Time parameters
    ramp_time = 5e-10  # 10 nanoseconds
    relaxation_time = 1.0e-8  # 10 nanoseconds
    simulation_time = 1.5e-6  # 1 microsecond
    save_interval = 3e-9  # 10 nanoseconds
    save_frequency = int(save_interval / dt)

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
        int((-0.025 - origin[0]) / spacing[0]),
        int((-0.025 - origin[1]) / spacing[1]),
        int((-0.005 - origin[2]) / spacing[2]),
    )
    hydro_shape = (
        int((0.05 / spacing[0]) + 0.5),
        int((0.05 / spacing[1]) + 0.5),
        int((0.06 / spacing[2]) + 0.5),
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

    # Make the mesh to id field operator
    mesh_to_id_field = MeshToIdField()

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
    ion_euler_update = IonEulerUpdate(
        apply_boundary_conditions_p7=apply_boundary_conditions_p7,
        apply_boundary_conditions_faces=apply_boundary_conditions_faces,
        emission_factor=0.5 * (electron_mass / proton_mass),
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

    # Initialize with id 1
    fill_id_field = FillIdField()
    id_field = fill_id_field(
        id_field,
        1,
    )

    # Get all parts and materials
    current_material_id = 2
    parts = list(findall(reactor, filter_=lambda node: isinstance(node, Part)))
    for i, part in tqdm(list(enumerate(parts)), desc="Voxelizing parts"):

        # Move part to global coordinates
        part_parent = part.parent
        while part_parent is not None:
            part = part_parent.location * part
            part_parent = part_parent.parent

        # Export part to stl
        name = f"{output_dir}/part_{i}.stl"
        part.export_stl(name)
        mesh = trimesh.load(name, file_type="stl", process=False)
        vertices = mesh.vertices
        indices = np.arange(vertices.shape[0])
        mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3),
            indices=wp.array(indices, dtype=wp.int32),
        )

        # Get material 
        if part.material.name is not "Vacuum":

            eps_mapping[current_material_id] = part.material.eps
            mu_mapping[current_material_id] = part.material.mu
            sigma_e_mapping[current_material_id] = part.material.sigma_e
            sigma_h_mapping[current_material_id] = part.material.sigma_m
            if part.material.initial_e is not None:
                initial_e_mapping[current_material_id] = part.material.initial_e

            # Voxelization
            id_field = mesh_to_id_field(
                mesh,
                id_field,
                current_material_id,
            )

            # Increment the material id
            current_material_id += 1

        else:
            # Voxelization Vacuum
            id_field = mesh_to_id_field(
                mesh,
                id_field,
                0,
            )

    # Save the id field
    save_to_temp_and_copy(
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
    save_to_temp_and_copy(
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
                direction=0,
            )

        # Call graph
        wp.capture_launch(capture.graph)

        # Check if nan
        if i % 1000 == 0:
            if np.isnan(np.sum(electric_field.data.numpy())):
                print("Electric field is NaN")
                exit()

    # Save the fields
    save_to_temp_and_copy(
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
                direction=0,
            )

        # Call graph
        wp.capture_launch(capture.graph)

    # Save the fields
    save_to_temp_and_copy(
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
        mass_i, momentum_i, energy_i = ion_euler_update(
            density_i,
            velocity_i,
            pressure_i,
            mass_i,
            momentum_i,
            energy_i,
            mass_e,
            momentum_e,
            energy_e,
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
            save_to_temp_and_copy(
                {
                    "id_field": id_field,
                    "electric_field": electric_field,
                    "magnetic_field": magnetic_field,
                },
                f"{output_dir}/simulation_em_field_{str(i).zfill(10)}.vtk",
            )
            save_to_temp_and_copy(
                {
                    "density_i": density_i,
                    "velocity_i": velocity_i,
                    "mass_i": mass_i,
                    "energy_i": energy_i,
                    "density_e": density_e,
                    "velocity_e": velocity_e,
                    "mass_e": mass_e,
                    "energy_e": energy_e,
                    "impressed_current": impressed_current,
                },
                f"{output_dir}/simulation_hydro_field_{str(i).zfill(10)}.vtk",
            )
