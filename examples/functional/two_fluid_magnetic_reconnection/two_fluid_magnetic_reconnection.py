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
        ion_mass: wp.float32,
        ion_charge: wp.float32,
        electron_mass: wp.float32,
        electron_charge: wp.float32,
    ):

        # Get index
        i, j, k = wp.tid()

        # Set top and bottom field ids
        #if j == 0:
        #    id_field.data[0, i, j, k] = wp.uint8(1)
        #if j == density_i.shape[1] - 1:
        #    id_field.data[0, i, j, k] = wp.uint8(2)

        # Get x, y, z
        x = density_i.origin[0] + wp.float32(i) * density_i.spacing[0] + 0.5 * density_i.spacing[0]
        y = density_i.origin[1] + wp.float32(j) * density_i.spacing[1] + 0.5 * density_i.spacing[1]
        z = density_i.origin[2] + wp.float32(k) * density_i.spacing[2] + 0.5 * density_i.spacing[2]

        # Constants
        b_0 = 0.1
        lambda_0 = 0.5
        n_0 = 1.0

        # Get initial magnetic field
        b_x = b_0 * (wp.tanh((z + 2.0 * wp.pi) / lambda_0) - wp.tanh((z - 2.0 * wp.pi) / lambda_0) - 1.0)
        b_y = 0.0
        b_z = 0.0

        # Get initial current density
        j_x = 0.0
        j_y = 1.0 * (b_0 / lambda_0) * ((1.0 / wp.cosh((z + 2.0 * wp.pi) / lambda_0)) ** 2.0 - (1.0 / wp.cosh((z - 2.0 * wp.pi) / lambda_0)) ** 2.0)
        j_z = 0.0

        # Number density
        n = n_0 * ((1.0 / 5.0) + ((1.0 / wp.cosh((z + 2.0 * wp.pi) / lambda_0)) ** 2.0) + ((1.0 / wp.cosh((z - 2.0 * wp.pi) / lambda_0)) ** 2.0))

        # Pressure
        #p = (b_0 / 12.0) * n + 0.5 * b_x ** 2.0
        p_e = 0.02
        p_i = 0.1

        # Set Density
        density_i.data[0, i, j, k] = n * ion_mass
        density_e.data[0, i, j, k] = n * electron_mass

        # Set Velocity
        velocity_i.data[0, i, j, k] = 0.0
        velocity_i.data[1, i, j, k] = 0.0
        velocity_i.data[2, i, j, k] = 0.0
        velocity_e.data[0, i, j, k] = j_x / (n * electron_charge)
        velocity_e.data[1, i, j, k] = j_y / (n * electron_charge)
        velocity_e.data[2, i, j, k] = j_z / (n * electron_charge)

        # Set Pressure
        pressure_i.data[0, i, j, k] = p_i - 0.002 * wp.sin(x/4.0)
        pressure_e.data[0, i, j, k] = p_e

        # Set Electric Field
        electric_field.data[0, i, j, k] = 0.0
        electric_field.data[1, i, j, k] = 0.0
        electric_field.data[2, i, j, k] = 0.0

        # Set Magnetic Field
        magnetic_field.data[0, i, j, k] = b_x
        magnetic_field.data[1, i, j, k] = b_y
        magnetic_field.data[2, i, j, k] = b_z


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
        ion_mass: float,
        ion_charge: float,
        electron_mass: float,
        electron_charge: float,
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
                ion_mass,
                ion_charge,
                electron_mass,
                electron_charge,
            ],
            dim=density_i.shape,
        )

        return density_i, velocity_i, pressure_i, density_e, velocity_e, pressure_e, electric_field, magnetic_field



if __name__ == '__main__':

    # Define simulation parameters
    dx = 0.05
    l_x = 8.0 * np.pi
    l_z = 8.0 * np.pi
    origin = (-l_x / 2.0, 0.0, -l_z / 2.0)
    spacing = (dx, 1.0, dx)
    shape = (int(l_x / dx), 1, int(l_z / dx))
    nr_cells = shape[0] * shape[1] * shape[2]
    simulation_time = 10.0
    save_frequency = 0.001
    gamma = (5.0 / 3.0)
    courant_factor = 0.2
    dt = courant_factor * dx / (wp.sqrt(3.0) * 1.0)
    electron_mass = 1.0
    proton_mass = 25.0
    electron_charge = -1.0
    proton_charge = 1.0

    # Make output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Make boundary conditions
    @wp.func
    def apply_boundary_conditions_p7(
        rho_stencil: p7_float32_stencil_type,
        vx_stencil: p7_float32_stencil_type,
        vy_stencil: p7_float32_stencil_type,
        vz_stencil: p7_float32_stencil_type,
        p_stencil: p7_float32_stencil_type,
        id_stencil: p7_uint8_stencil_type,
    ):
    
        # Apply boundary conditions
        for i in range(6):

            # stencil index
            st_idx = i + 1
    
            # Check if on the boundary
            if id_stencil[st_idx] == wp.uint8(1) or id_stencil[st_idx] == wp.uint8(2):

                rho_stencil[st_idx] = rho_stencil[0]
                vx_stencil[st_idx] = vx_stencil[0]
                vy_stencil[st_idx] = vy_stencil[0]
                vz_stencil[st_idx] = vz_stencil[0]
                p_stencil[st_idx] = p_stencil[0]

        return rho_stencil, vx_stencil, vy_stencil, vz_stencil, p_stencil

    @wp.func
    def apply_boundary_conditions_faces(
        rho_faces: faces_float32_type,
        vx_faces: faces_float32_type,
        vy_faces: faces_float32_type,
        vz_faces: faces_float32_type,
        p_faces: faces_float32_type,
        rho_stencil: p4_float32_stencil_type,
        vx_stencil: p4_float32_stencil_type,
        vy_stencil: p4_float32_stencil_type,
        vz_stencil: p4_float32_stencil_type,
        p_stencil: p4_float32_stencil_type,
        id_stencil: p4_uint8_stencil_type,
    ):

        # Apply boundary conditions
        for i in range(3):

            # Check if on the boundary
            if id_stencil[0] == wp.uint8(1) or id_stencil[i + 1] == wp.uint8(1) or id_stencil[0] == wp.uint8(2) or id_stencil[i + 1] == wp.uint8(2):

                # Check boundary condition
                # Outlet
                if id_stencil[0] == wp.uint8(1) or id_stencil[0] == wp.uint8(2):
                    rho_faces[2 * i] = rho_stencil[i + 1]
                    vx_faces[2 * i] = vx_stencil[i + 1]
                    vy_faces[2 * i] = vy_stencil[i + 1]
                    vz_faces[2 * i] = vz_stencil[i + 1]
                    p_faces[2 * i] = p_stencil[i + 1]
                if id_stencil[i + 1] == wp.uint8(1) or id_stencil[i + 1] == wp.uint8(2):
                    rho_faces[2 * i + 1] = rho_stencil[0]
                    vx_faces[2 * i + 1] = vx_stencil[0]
                    vy_faces[2 * i + 1] = vy_stencil[0]
                    vz_faces[2 * i + 1] = vz_stencil[0]
                    p_faces[2 * i + 1] = p_stencil[0]

        return rho_faces, vx_faces, vy_faces, vz_faces, p_faces
 
    @wp.func
    def magnetic_boundary_conditions(
        id_0_0_1: wp.uint8,
        id_0_1_0: wp.uint8,
        id_0_1_1: wp.uint8,
        id_1_0_0: wp.uint8,
        id_1_0_1: wp.uint8,
        id_1_1_0: wp.uint8,
        id_1_1_1: wp.uint8,
        m_x_1_1_1: wp.float32,
        m_x_1_0_1: wp.float32,
        m_x_1_1_0: wp.float32,
        m_y_1_1_1: wp.float32,
        m_y_0_1_1: wp.float32,
        m_y_1_1_0: wp.float32,
        m_z_1_1_1: wp.float32,
        m_z_0_1_1: wp.float32,
        m_z_1_0_1: wp.float32,
    ):
        if id_0_1_1 == wp.uint8(1):
            m_y_0_1_1 = 0.0
            m_z_0_1_1 = 0.0
        if id_1_0_1 == wp.uint8(1):
            m_x_1_0_1 = -0.1
            m_z_1_0_1 = 0.0
        if id_1_1_0 == wp.uint8(1):
            m_x_1_1_0 = -0.1
            m_y_1_1_0 = 0.0
        if id_0_1_1 == wp.uint8(2):
            m_y_0_1_1 = 0.0
            m_z_0_1_1 = 0.0
        if id_1_0_1 == wp.uint8(2):
            m_x_1_0_1 = 0.1
            m_z_1_0_1 = 0.0
        if id_1_1_0 == wp.uint8(2):
            m_x_1_1_0 = 0.1
            m_y_1_1_0 = 0.0
        return (
            m_x_1_1_1,
            m_x_1_0_1,
            m_x_1_1_0,
            m_y_1_1_1,
            m_y_0_1_1,
            m_y_1_1_0,
            m_z_1_1_1,
            m_z_0_1_1,
            m_z_1_0_1,
        )

    @wp.func
    def electric_boundary_conditions(
        id_1_1_1: wp.uint8,
        id_0_1_1: wp.uint8,
        id_1_0_1: wp.uint8,
        id_1_1_0: wp.uint8,
        e_x_0_0_0: wp.float32,
        e_x_0_1_0: wp.float32,
        e_x_0_0_1: wp.float32,
        e_y_0_0_0: wp.float32,
        e_y_1_0_0: wp.float32,
        e_y_0_0_1: wp.float32,
        e_z_0_0_0: wp.float32,
        e_z_1_0_0: wp.float32,
        e_z_0_1_0: wp.float32,
    ):
        return (
            e_x_0_0_0,
            e_x_0_1_0,
            e_x_0_0_1,
            e_y_0_0_0,
            e_y_1_0_0,
            e_y_0_0_1,
            e_z_0_0_0,
            e_z_1_0_0,
            e_z_0_1_0,
        )

    # Make the constructor
    constructor = Constructor()

    # Make the operators
    initialize_magnetic_reconnection = InitializeMagneticReconnection()
    primitive_to_conservative = PrimitiveToConservative()
    conservative_to_primitive = ConservativeToPrimitive()
    get_time_step = GetTimeStep()
    euler_update = EulerUpdate(
        apply_boundary_conditions_p7=apply_boundary_conditions_p7,
        apply_boundary_conditions_faces=apply_boundary_conditions_faces,
    )
    add_em_source_terms = AddEMSourceTerms()
    get_current_density = GetCurrentDensity()
    yee_electric_field_update = YeeElectricFieldUpdate()
    yee_magnetic_field_update = YeeMagneticFieldUpdate(apply_boundary_conditions=magnetic_boundary_conditions)
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
        proton_mass,
        proton_charge,
        electron_mass,
        electron_charge,
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
                proton_mass,
                proton_charge,
                1.0,
            )
            current_density = get_current_density(
                density_e,
                velocity_e,
                pressure_e,
                current_density,
                electron_mass,
                electron_charge,
                1.0,
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
                proton_charge,
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
                electron_charge,
                dt,
            )

            # Check if time passes save frequency
            remander = current_time % save_frequency 
            if (remander + dt) > save_frequency:
                save_index += 1
                print(f"Saved {save_index} files")
                print(dt)
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
                    os.path.join(output_dir, f"t_{str(save_index).zfill(4)}.vtk")
                )

            # Compute MUPS
            if nr_iterations % 10 == 0:
                wp.synchronize()
                toc = time.time()
                mups = nr_cells * nr_iterations / (toc - tic) / 1.0e6
                print(f"Elapsed time: {toc - tic}")
                print(f"Iterations: {nr_iterations}")
                print(f"MUPS: {mups}")

            # Update the time
            current_time += dt

            # Update the progress bar
            pbar.update(dt)

            # Update the number of iterations
            nr_iterations += 1
