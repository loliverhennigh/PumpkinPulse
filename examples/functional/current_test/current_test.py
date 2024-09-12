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

class BlastInitializer(Operator):

    @wp.kernel
    def _initialize_blast(
        density_i: Fieldfloat32,
        velocity_i: Fieldfloat32,
        pressure_i: Fieldfloat32,
        density_e: Fieldfloat32,
        velocity_e: Fieldfloat32,
        pressure_e: Fieldfloat32,
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
        id_field: Fielduint8,
        proton_mass: wp.float32,
        electron_mass: wp.float32,
    ):

        # Get index
        i, j, k = wp.tid()

        # Get x, y, z
        x = density_i.origin[0] + wp.float32(i) * density_i.spacing[0]
        y = density_i.origin[1] + wp.float32(j) * density_i.spacing[1]
        z = density_i.origin[2] + wp.float32(k) * density_i.spacing[2]

        # Parameters
        length = 1.0
        radius = 0.1
        cross_section = wp.pi * radius**2.0
        nr_electrons = 6.242197253433209e+18
        current_density = 1.0
        number_density = nr_electrons / (length * cross_section)

        # Set density
        distance = wp.sqrt(x**2.0 + y**2.0)

        # Set density
        density_i.data[0, i, j, k] = number_density * proton_mass
        density_e.data[0, i, j, k] = number_density * electron_mass

        # Set pressure
        pressure_i.data[0, i, j, k] = 1.0
        pressure_e.data[0, i, j, k] = 1.0

        # Set velocity
        if distance < radius:
            velocity_i.data[0, i, j, k] = 0.0
            velocity_i.data[1, i, j, k] = 0.0
            velocity_i.data[2, i, j, k] = 0.0
            velocity_e.data[0, i, j, k] = 0.0
            velocity_e.data[1, i, j, k] = 0.0
            velocity_e.data[2, i, j, k] = current_density * cross_section
        else:
            velocity_i.data[0, i, j, k] = 0.0
            velocity_i.data[1, i, j, k] = 0.0
            velocity_i.data[2, i, j, k] = 0.0
            velocity_e.data[0, i, j, k] = 0.0
            velocity_e.data[1, i, j, k] = 0.0
            velocity_e.data[2, i, j, k] = 0.0



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
        proton_mass: float,
        electron_mass: float,
    ):
        # Launch kernel
        wp.launch(
            self._initialize_blast,
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
                proton_mass,
                electron_mass,
            ],
            dim=density_i.shape,
        )

        return density_i, velocity_i, pressure_i, density_e, velocity_e, pressure_e, electric_field, magnetic_field



if __name__ == '__main__':

    # Geometry parameters
    dx = 0.01 # m
    l = 1.0
    origin = (-l / 2.0, -l / 2.0, -l / 2.0)
    spacing = (dx, dx, dx)
    shape = (int(l / dx), int(l / dx), int(l / dx))
    nr_cells = shape[0] * shape[1] * shape[2]

    # Electromagnetic Constants
    elementary_charge = 1.60217662e-19
    epsilon_0 = 8.85418782e-12
    mu_0 = 4.0 * 3.14159 * 1.0e-7
    eta_0 = np.sqrt(mu_0 / epsilon_0)
    c = (1.0 / np.sqrt(mu_0 * epsilon_0))
    electron_mass = 9.10938356e-31
    proton_mass = 1.6726219e-27

    # Fluid Constants
    gamma = (5.0 / 3.0)

    # Plasma ball parameters
    blast_number_density = 5.0e17
    background_number_density = 5.0e16
    blast_pressure = 100.0
    background_pressure = 1.0
    plasma_radius = 0.4
    plasma_position = (0.0, 0.0, 0.0)

    # Time parameters
    simulation_time = 1e-6
    save_frequency = 5e-10
    courant_factor = 0.4
    em_dt = courant_factor * (1.0 / (c * np.sqrt(3.0 / (dx ** 2.0))))

    # Make output directory
    output_dir = f"output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Make the constructor
    constructor = Constructor()

    # Make the operators
    initialize_blast = BlastInitializer()
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
    eps_mapping = wp.from_numpy(np.array([epsilon_0], dtype=np.float32), dtype=wp.float32)
    mu_mapping = wp.from_numpy(np.array([mu_0], dtype=np.float32), dtype=wp.float32)
    sigma_e_mapping = wp.from_numpy(np.array([0.0], dtype=np.float32), dtype=wp.float32)
    sigma_m_mapping = wp.from_numpy(np.array([0.0], dtype=np.float32), dtype=wp.float32)

    # Initialize the fields
    (
        density_i,
        velocity_i,
        pressure_i,
        density_e,
        velocity_e,
        pressure_e,
        electric_field,
        magnetic_field,
    ) = initialize_blast(
        density_i,
        velocity_i,
        pressure_i,
        density_e,
        velocity_e,
        pressure_e,
        electric_field,
        magnetic_field,
        id_field,
        proton_mass,
        electron_mass,
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

    # Get the current density
    current_density.data.zero_()
    current_density = get_current_density(
        density_i,
        velocity_i,
        pressure_i,
        current_density,
        proton_mass,
        elementary_charge,
        1.0,
    )
    current_density = get_current_density(
        density_e,
        velocity_e,
        pressure_e,
        current_density,
        electron_mass,
        -elementary_charge,
        1.0,
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
    exit()

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

            # Get the time step
            dt_i = get_time_step(
                density_i,
                velocity_i,
                pressure_i,
                id_field,
                courant_factor,
                gamma,
            )
            dt_e = get_time_step(
                density_e,
                velocity_e,
                pressure_e,
                id_field,
                courant_factor,
                gamma,
            )
            dt = min(dt_i, dt_e, em_dt)
            #print(f"Time Step EM: {em_dt}")
            #print(f"Time Step Ion: {dt_i}")
            #print(f"Time Step Elec: {dt_e}")
            #print(f"Time Step: {dt}")

            # Get the current density
            current_density.data.zero_()
            current_density = get_current_density(
                density_i,
                velocity_i,
                pressure_i,
                current_density,
                proton_mass,
                elementary_charge,
                mu_0,
            )
            current_density = get_current_density(
                density_e,
                velocity_e,
                pressure_e,
                current_density,
                electron_mass,
                -elementary_charge,
                mu_0,
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
                proton_mass,
                elementary_charge,
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
                electron_mass,
                -elementary_charge,
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
                        #"velocity_i": velocity_i,
                        "pressure_i": pressure_i,
                        "density_e": density_e,
                        #"velocity_e": velocity_e,
                        "pressure_e": pressure_e,
                        "electric_field": electric_field,
                        #"magnetic_field": magnetic_field,
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
