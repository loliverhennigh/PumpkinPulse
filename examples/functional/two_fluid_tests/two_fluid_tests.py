# Implosion of a pumpkin cavity

import os
import numpy as np
import warp as wp
from tqdm import tqdm
import time

wp.init()

from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.data import Fieldfloat32, Fielduint8
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
    YeeMagneticFieldUpdate
)
from pumpkin_pulse.operator.saver import FieldSaver
from pumpkin_pulse.functional.stencil import (
    p7_float32_stencil_type,
    p7_uint8_stencil_type,
    p4_float32_stencil_type,
    p4_uint8_stencil_type,
    faces_float32_type,
)

class InitializeMagneticReconnection(Operator):

    @wp.kernel
    def _initialize_z_pinch(
        density_i: Fieldfloat32,
        velocity_i: Fieldfloat32,
        pressure_i: Fieldfloat32,
        density_e: Fieldfloat32,
        velocity_e: Fieldfloat32,
        pressure_e: Fieldfloat32,
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
        id_field: Fielduint8,
        species_mass_i: wp.float32,
        charge_i: wp.float32,
        species_mass_e: wp.float32,
        charge_e: wp.float32,
    ):

        # Get index
        i, j, k = wp.tid()

        # Get x, y, z
        x = density_i.origin[0] + wp.float32(i) * density_i.spacing[0] + 0.5 * density_i.spacing[0]
        y = density_i.origin[1] + wp.float32(j) * density_i.spacing[1] + 0.5 * density_i.spacing[1]
        z = density_i.origin[2] + wp.float32(k) * density_i.spacing[2] + 0.5 * density_i.spacing[2]
        bx_y = y
        jz_y = y

        # Constants
        b_0 = 0.1
        lambda_0 = 0.2
        n_0 = 1.0
        phi_0 = b_0 / 10.0
        l = 8.0 * wp.pi

        # Get initial magnetic field
        b_x = b_0 * (wp.tanh((bx_y + 2.0 * wp.pi) / lambda_0) - wp.tanh((bx_y - 2.0 * wp.pi) / lambda_0) - 1.0)
        b_y = 0.0
        b_z = 0.0

        # Get initial current density
        j_x = 0.0
        j_y = 0.0
        j_z = - (b_0 / lambda_0) * ((1.0 / wp.cosh((jz_y + 2.0 * wp.pi) / lambda_0)) ** 2.0 - (1.0 / wp.cosh((jz_y - 2.0 * wp.pi) / lambda_0)) ** 2.0)

        # Number density
        n = n_0 * ((1.0 / 5.0) + ((1.0 / wp.cosh((y + 2.0 * wp.pi) / lambda_0)) ** 2.0) + ((1.0 / wp.cosh((y - 2.0 * wp.pi) / lambda_0)) ** 2.0))

        # Pressure
        p = (b_0 / 20.0) * n
        p_e = p
        p_i = 5.0 * (b_0 / 20.0)

        # Get b_pert
        b_pert_x = phi_0 * (2.0 * wp.pi / l) * wp.sin(2.0 * wp.pi * y / l) * wp.cos(2.0 * wp.pi * x / l)
        b_pert_y = - phi_0 * (2.0 * wp.pi / l) * wp.sin(2.0 * wp.pi * x / l) * wp.cos(2.0 * wp.pi * y / l)
        b_pert_z = 0.0

        # Set Density
        density_i.data[0, i, j, k] = n * species_mass_i
        density_e.data[0, i, j, k] = n * species_mass_e

        # Set Velocity
        velocity_i.data[0, i, j, k] = 0.0
        velocity_i.data[1, i, j, k] = 0.0
        velocity_i.data[2, i, j, k] = 0.0
        velocity_e.data[0, i, j, k] = j_x / (n * charge_e)
        velocity_e.data[1, i, j, k] = j_y / (n * charge_e)
        velocity_e.data[2, i, j, k] = j_z / (n * charge_e)

        # Set Pressure
        pressure_i.data[0, i, j, k] = p_i
        pressure_e.data[0, i, j, k] = p_e

        # Set Electric Field
        electric_field.data[0, i, j, k] = 0.0
        electric_field.data[1, i, j, k] = 0.0
        electric_field.data[2, i, j, k] = 0.0

        # Set Magnetic Field
        magnetic_field.data[0, i, j, k] = b_x + 1.0 * b_pert_x
        magnetic_field.data[1, i, j, k] = b_y + 1.0 * b_pert_y
        magnetic_field.data[2, i, j, k] = b_z + 1.0 * b_pert_z


    def __call__(
        self,
        density_i: Fieldfloat32,
        velocity_i: Fieldfloat32,
        pressure_i: Fieldfloat32,
        density_e: Fieldfloat32,
        velocity_e: Fieldfloat32,
        pressure_e: Fieldfloat32,
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
        id_field: Fielduint8,
        species_mass_i: wp.float32,
        charge_i: wp.float32,
        species_mass_e: wp.float32,
        charge_e: wp.float32,
    ):
        # Launch kernel
        wp.launch(
            self._initialize_z_pinch,
            inputs=[
                density_i,
                velocity_i,
                pressure_i,
                density_e,
                velocity_e,
                pressure_e,
                electric_field,
                magnetic_field,
                id_field,
                species_mass_i,
                charge_i,
                species_mass_e,
                charge_e,
            ],
            dim=density_i.shape,
        )

        return density_i, velocity_i, pressure_i, density_e, velocity_e, pressure_e, electric_field, magnetic_field



if __name__ == '__main__':

    # Define simulation parameters
    dx = 0.01
    l_x = 8.0 * np.pi
    l_y = 8.0 * np.pi
    origin = (-l_x / 2.0, -l_y / 2.0, -dx / 2.0)
    spacing = (dx, dx, dx)
    shape = (int(l_x / dx), int(l_y / dx), 1)
    nr_cells = shape[0] * shape[1] * shape[2]
    simulation_time = 5000.0
    save_frequency = 2.5
    gamma = (5.0 / 3.0)
    courant_factor = 0.9
    dt = courant_factor * dx / (wp.sqrt(3.0) * 1.0)
    species_mass_i = 25.0
    charge_i = 1.0
    species_mass_e = 1.0
    charge_e = -1.0

    # Make output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Make the constructor
    constructor = Constructor()

    # Make the operators
    initialize_magnetic_reconnection = InitializeMagneticReconnection()
    primitive_to_conservative = PrimitiveToConservative()
    conservative_to_primitive = ConservativeToPrimitive()
    get_time_step = GetTimeStep()
    euler_update = EulerUpdate()
    add_em_source_terms = AddEMSourceTerms()
    get_current_density = GetCurrentDensity()
    yee_electric_field_update = YeeElectricFieldUpdate()
    yee_magnetic_field_update = YeeMagneticFieldUpdate()
    field_saver = FieldSaver()

    # Make the fields
    density_i = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    velocity_i = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    pressure_i = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    density_e = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    velocity_e = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    pressure_e = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    mass_i = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    momentum_i = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    energy_i = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    mass_e = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    momentum_e = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    energy_e = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    electric_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    magnetic_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    current_density = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    id_field = constructor.create_field(
        dtype=wp.uint8,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    eps_mapping = wp.from_numpy(np.array([1.0], dtype=np.float32), dtype=wp.float32)
    mu_mapping = wp.from_numpy(np.array([1.0], dtype=np.float32), dtype=wp.float32)
    sigma_e_mapping = wp.from_numpy(np.array([0.0], dtype=np.float32), dtype=wp.float32)
    sigma_m_mapping = wp.from_numpy(np.array([0.0], dtype=np.float32), dtype=wp.float32)

    # Initialize the fields
    density_i, velocity_i, pressure_i, density_e, velocity_e, pressure_e, electric_field, magnetic_field = initialize_magnetic_reconnection(
        density_i,
        velocity_i,
        pressure_i,
        density_e,
        velocity_e,
        pressure_e,
        electric_field,
        magnetic_field,
        id_field,
        species_mass_i,
        charge_i,
        species_mass_e,
        charge_e,
    )

    # Get conservative variables
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

    # Save the fields
    field_saver(
        {
            "density_i": density_i,
            "velocity_i": velocity_i,
            "pressure_i": pressure_i,
            "density_e": density_e,
            "velocity_e": velocity_e,
            "pressure_e": pressure_e,
            "electric_field": electric_field,
            "magnetic_field": magnetic_field,
            "current_density": current_density,
        },
        os.path.join(output_dir, "initial_conditions.vtk")
    )

    # Run the simulation
    current_time = 0.0
    save_index = 0
    nr_iterations = 0
    tic = time.time()
    with tqdm(total=simulation_time, desc="Simulation Progress") as pbar:
        while current_time < simulation_time:

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

            # Get the current density
            current_density.data.zero_()
            current_density = get_current_density(
                density_i,
                velocity_i,
                pressure_i,
                current_density,
                species_mass_i,
                charge_i,
                1.0,
            )
            current_density = get_current_density(
                density_e,
                velocity_e,
                pressure_e,
                current_density,
                species_mass_e,
                charge_e,
                1.0,
            )

            ## Get the time step
            #dt = get_time_step(
            #    density,
            #    velocity,
            #    pressure,
            #    cell_magnetic_field,
            #    id_field,
            #    courant_factor,
            #    gamma,
            #)

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

            # Update the magnetic field
            magnetic_field = yee_magnetic_field_update(
                electric_field,
                magnetic_field,
                id_field,
                mu_mapping,
                sigma_m_mapping,
                dt
            )

            # Update the electric
            electric_field = yee_electric_field_update(
                electric_field,
                magnetic_field,
                current_density,
                id_field,
                eps_mapping,
                sigma_e_mapping,
                dt
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
                species_mass_i,
                charge_i,
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
                species_mass_e,
                charge_e,
                dt,
            )

            # Check if time passes save frequency
            remander = current_time % save_frequency 
            if (remander + dt) > save_frequency:
                save_index += 1
                print(f"Saved {save_index} files")
                field_saver(
                    {
                        "density_i": density_i,
                        "velocity_i": velocity_i,
                        "pressure_i": pressure_i,
                        "density_e": density_e,
                        "velocity_e": velocity_e,
                        "pressure_e": pressure_e,
                        #"mass_i": mass_i,
                        #"mass_e": mass_e,
                        #"momentum_i": momentum_i,
                        #"momentum_e": momentum_e,
                        #"energy_i": energy_i,
                        #"energy_e": energy_e,
                        "electric_field": electric_field,
                        "magnetic_field": magnetic_field,
                        #"current_density": current_density,
                    },
                    os.path.join(output_dir, f"t_{str(save_index).zfill(4)}.vtk")
                )
                #if save_index == 10:
                #    exit()

            # Compute MUPS
            if nr_iterations % 10 == 0:
                wp.synchronize()
                toc = time.time()
                mups = nr_cells * nr_iterations / (toc - tic) / 1.0e6
                print(f"Iterations: {nr_iterations}")
                print(f"MUPS: {mups}")

            # Update the time
            current_time += dt

            # Update the progress bar
            pbar.update(dt)

            # Update the number of iterations
            nr_iterations += 1
