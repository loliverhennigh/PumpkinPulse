# Euler simulation of De Laval nozzle for rocket propulsion

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
from pumpkin_pulse.operator.mesh import StlToMesh, MeshToIdField
from pumpkin_pulse.operator.saver import FieldSaver
from pumpkin_pulse.functional.stencil import (
    p7_float32_stencil_type,
    p7_uint8_stencil_type,
    p4_float32_stencil_type,
    p4_uint8_stencil_type,
    faces_float32_type,
)

####### STL file ########
# Ref: https://www.thingiverse.com/thing:280483
#License
#BSD-2-CLAUSE
#Customizable Rocket Nozzle
#by waterside is licensed under the BSD License license.
#########################

class InitializePrimitives(Operator):

    @wp.kernel
    def _initialize_vel(
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
        set_density: wp.float32,
        set_velocity: wp.vec3,
        set_pressure: wp.float32,
    ):

        # Get index
        i, j, k = wp.tid()

        # Initialize the density, velocity, and pressure
        density.data[0, i, j, k] = set_density
        velocity.data[0, i, j, k] = set_velocity[0]
        velocity.data[1, i, j, k] = set_velocity[1]
        velocity.data[2, i, j, k] = set_velocity[2]
        pressure.data[0, i, j, k] = set_pressure

    def __call__(
        self,
        density,
        velocity,
        pressure,
        set_density,
        set_velocity,
        set_pressure,
    ):
        # Launch kernel
        wp.launch(
            self._initialize_vel,
            inputs=[
                density,
                velocity,
                pressure,
                set_density,
                set_velocity,
                set_pressure,
            ],
            dim=density.shape,
        )

        return density, velocity, pressure


class InitializeBoundary(Operator):
    # 1: Outlet
    # 2: Slip wall

    @wp.kernel
    def _initialize_boundary(
        id_field: Fielduint8,
        center: wp.vec3,
        radius: wp.float32,
    ):
        # Get index
        i, j, k = wp.tid()

        # Get x, y, z
        x = id_field.origin[0] + wp.float32(i) * id_field.spacing[0]
        y = id_field.origin[1] + wp.float32(j) * id_field.spacing[1]
        z = id_field.origin[2] + wp.float32(k) * id_field.spacing[2]

        # Check if on any side but i = 0
        if j == 0 or k == 0 or j == id_field.shape[1] - 1 or k == id_field.shape[2] - 1 or i == id_field.shape[0] - 1:
            id_field.data[0, i, j, k] = wp.uint8(1)
        elif i == 0:
            # check if inside the circle
            if (y - center[1]) ** 2.0 + (z - center[2]) ** 2.0 < radius ** 2.0:
                id_field.data[0, i, j, k] = wp.uint8(2) # close exit
            else:
                id_field.data[0, i, j, k] = wp.uint8(1)
        else:
            return

    def __call__(
        self,
        id_field,
        center,
        radius,
    ):
        # Launch kernel
        wp.launch(
            self._initialize_boundary,
            inputs=[
                id_field,
                center,
                radius,
            ],
            dim=id_field.shape,
        )

        return id_field

class InjectFuel(Operator):

    @wp.kernel
    def _inject_fuel(
        mass: Fieldfloat32,
        energy: Fieldfloat32,
        id_field: Fielduint8,
        mass_flow_rate: wp.float32,
        energy_flow_rate: wp.float32,
        center: wp.vec3,
        radius: wp.float32,
        dt: wp.float32,
    ):
        # Get index
        i, j, k = wp.tid()

        # Check if on the boundary
        if id_field.data[0, i, j, k] != wp.uint8(0):
            return

        # Get x, y, z
        x = mass.origin[0] + wp.float32(i) * mass.spacing[0]
        y = mass.origin[1] + wp.float32(j) * mass.spacing[1]
        z = mass.origin[2] + wp.float32(k) * mass.spacing[2]

        # Get volume of cylinder
        volume = wp.pi * radius ** 2.0 * mass.spacing[0]

        # Get mass and energy flow rate over volume and time
        mass_flow_rate_volume = mass_flow_rate * dt / volume
        energy_flow_rate_volume = energy_flow_rate * dt / volume

        # Check if in a sphere
        if i == 1:
            if (y - center[1]) ** 2.0 + (z - center[2]) ** 2.0 < radius ** 2.0:

                # Current mass and energy
                current_mass = mass.data[0, i, j, k]
                current_energy = energy.data[0, i, j, k]

                # Update mass and energy
                mass.data[0, i, j, k] = current_mass + 1.00 * dt * mass.spacing[0] * mass.spacing[1] * mass.spacing[2]
                energy.data[0, i, j, k] = current_energy + 10.0 * dt * mass.spacing[0] * mass.spacing[1] * mass.spacing[2]


    def __call__(
        self,
        mass,
        energy,
        id_field,
        mass_flow_rate,
        energy_flow_rate,
        center,
        radius,
        dt,
    ):
        # Launch kernel
        wp.launch(
            self._inject_fuel,
            inputs=[
                mass,
                energy,
                id_field,
                mass_flow_rate,
                energy_flow_rate,
                center,
                radius,
                dt,
            ],
            dim=mass.shape,
        )

        return mass, energy


 
if __name__ == '__main__':

    # Define simulation parameters
    #dx = 0.0025
    dx = 0.005
    origin = (0.0, -0.4, -0.4)
    spacing = (dx, dx, dx)
    shape = (int(3.0/dx), int(0.8/dx), int(0.8/dx))
    nr_cells = shape[0] * shape[1] * shape[2]
    print("Number of cells: ", nr_cells)

    # Fluid parameters
    gamma = 1.4
    courant_factor = 0.4

    # Nozzle parameters
    stagnation_pressure = 5.0e6 # Pa
    stagnation_temperature = 3000.0 # K
    specific_heat = 1000.0 # J/(kg*K)
    mass_flow_rate = 1.0 # kg/s
    energy_flow_rate = stagnation_temperature * mass_flow_rate * specific_heat # W
    print("Mass flow rate: ", mass_flow_rate)
    print("Energy flow rate: ", energy_flow_rate)

    # Outlet parameters
    outlet_pressure = 1.0e5 # Pa (1 atm)
    outlet_density = 1.225 # kg/m^3 (1 atm)

    # Time parameters
    solve_time = 0.1 # s
    save_frequency = 0.0001 # s

    # Reference values for non-dimensionalization
    reference_density = outlet_density
    reference_pressure = outlet_pressure
    reference_velocity = np.sqrt(gamma * reference_pressure / reference_density)
    reference_length = 1.0
    reference_time = reference_length / reference_velocity
    reference_mass = reference_density * reference_length ** 3.0
    reference_energy = reference_mass * reference_velocity ** 2.0
    print("Reference density: ", reference_density)
    print("Reference pressure: ", reference_pressure)
    print("Reference velocity: ", reference_velocity)
    print("Reference length: ", reference_length)
    print("Reference time: ", reference_time)
    print("Reference mass: ", reference_mass)
    print("Reference energy: ", reference_energy)

    # Non-dimensionalization
    nondim_stagnation_pressure = stagnation_pressure / reference_pressure
    nondim_outlet_density = outlet_density / reference_density
    nondim_outlet_pressure = outlet_pressure / reference_pressure
    nondim_solve_time = solve_time / reference_time
    nondim_save_frequency = save_frequency / reference_time
    nondim_mass_flow_rate = mass_flow_rate / (reference_mass / reference_time)
    nondim_energy_flow_rate = energy_flow_rate / (reference_energy / reference_time)
    print("Non-dimensional stagnation pressure: ", nondim_stagnation_pressure)
    print("Non-dimensional outlet density: ", nondim_outlet_density)
    print("Non-dimensional outlet pressure: ", nondim_outlet_pressure)
    print("Non-dimensional solve time: ", nondim_solve_time)
    print("Non-dimensional save frequency: ", nondim_save_frequency)
    print("Non-dimensional mass flow rate: ", nondim_mass_flow_rate)
    print("Non-dimensional energy flow rate: ", nondim_energy_flow_rate)

    # Make output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Make the constructor
    constructor = Constructor()

    # Make boundary conditions functions
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
            if id_stencil[st_idx] != wp.uint8(0):

                # Check boundary condition
                # Outlet
                if id_stencil[st_idx] == wp.uint8(1):
                    rho_stencil[st_idx] = rho_stencil[0]
                    vx_stencil[st_idx] = vx_stencil[0]
                    vy_stencil[st_idx] = vy_stencil[0]
                    vz_stencil[st_idx] = vz_stencil[0]
                    p_stencil[st_idx] = p_stencil[0]

                # Wall
                if id_stencil[st_idx] == wp.uint8(2):

                    # Get normal
                    if st_idx == 1 or st_idx == 2:
                        flip_vx = -1.0
                    else:
                        flip_vx = 1.0
                    if st_idx == 3 or st_idx == 4:
                        flip_vy = -1.0
                    else:
                        flip_vy = 1.0
                    if st_idx == 5 or st_idx == 6:
                        flip_vz = -1.0
                    else:
                        flip_vz = 1.0

                    # Apply wall boundary condition
                    rho_stencil[st_idx] = rho_stencil[0]
                    vx_stencil[st_idx] = vx_stencil[0] * flip_vx
                    vy_stencil[st_idx] = vy_stencil[0] * flip_vy
                    vz_stencil[st_idx] = vz_stencil[0] * flip_vz
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
            if id_stencil[0] != wp.uint8(0) or id_stencil[i + 1] != wp.uint8(0):

                # Check boundary condition
                # Outlet
                if id_stencil[0] == wp.uint8(1):
                    rho_faces[2 * i] = rho_stencil[i + 1]
                    vx_faces[2 * i] = vx_stencil[i + 1]
                    vy_faces[2 * i] = vy_stencil[i + 1]
                    vz_faces[2 * i] = vz_stencil[i + 1]
                    p_faces[2 * i] = p_stencil[i + 1]
                if id_stencil[i + 1] == wp.uint8(1):
                    rho_faces[2 * i + 1] = rho_stencil[0]
                    vx_faces[2 * i + 1] = vx_stencil[0]
                    vy_faces[2 * i + 1] = vy_stencil[0]
                    vz_faces[2 * i + 1] = vz_stencil[0]
                    p_faces[2 * i + 1] = p_stencil[0]

                # Get normal
                if i == 0:
                    flip_vx, flip_vy, flip_vz = -1.0, 1.0, 1.0
                if i == 1:
                    flip_vx, flip_vy, flip_vz = 1.0, -1.0, 1.0
                if i == 2:
                    flip_vx, flip_vy, flip_vz = 1.0, 1.0, -1.0

                # Wall
                if id_stencil[0] == wp.uint8(2):
                    rho_faces[2 * i] = rho_stencil[i + 1]
                    vx_faces[2 * i] = vx_stencil[i + 1]
                    vy_faces[2 * i] = vy_stencil[i + 1]
                    vz_faces[2 * i] = vz_stencil[i + 1]
                    p_faces[2 * i] = p_stencil[i + 1]
                if id_stencil[i + 1] == wp.uint8(2):
                    rho_faces[2 * i + 1] = rho_stencil[0]
                    vx_faces[2 * i + 1] = vx_stencil[0] * flip_vx
                    vy_faces[2 * i + 1] = vy_stencil[0] * flip_vy
                    vz_faces[2 * i + 1] = vz_stencil[0] * flip_vz
                    p_faces[2 * i + 1] = p_stencil[0]

        return rho_faces, vx_faces, vy_faces, vz_faces, p_faces

    # Make the operators
    initialize_primitives = InitializePrimitives()
    initialize_boundary = InitializeBoundary()
    inject_fuel = InjectFuel()
    field_saver = FieldSaver()
    stl_to_mesh = StlToMesh()
    mesh_to_id_field = MeshToIdField()
    primitive_to_conservative = PrimitiveToConservative()
    conservative_to_primitive = ConservativeToPrimitive()
    get_time_step = GetTimeStep()
    euler_update = EulerUpdate(
        apply_boundary_conditions_p7=apply_boundary_conditions_p7,
        apply_boundary_conditions_faces=apply_boundary_conditions_faces,
    )

    # Make the fields
    id_field = constructor.create_field(
        dtype=wp.uint8,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    density = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    velocity = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    pressure = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    mass = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    momentum = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    energy = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )

    # Set the boundary conditions
    id_field = initialize_boundary(
        id_field,
        center=(0.0, 0.0, 0.0),
        radius=0.0875,
    )

    # Make mesh
    mesh = stl_to_mesh("./files/de_laval_nozzle.stl")
    id_field = mesh_to_id_field(mesh, id_field, 2)

    # Save id field
    field_saver(id_field, os.path.join(output_dir, "id_field.vtk"))

    # Initialize the primitives
    density, velocity, pressure = initialize_primitives(
        density,
        velocity,
        pressure,
        set_density=nondim_outlet_density,
        set_velocity=(0.0, 0.0, 0.0),
        set_pressure=nondim_outlet_pressure,
    )

    # Convert primitives to conservative
    mass, momentum, energy = primitive_to_conservative(
        density,
        velocity,
        pressure,
        mass,
        momentum,
        energy,
        gamma
    )

    # Save initial conditions
    field_saver(
        {"density": density, "velocity": velocity, "pressure": pressure},
        os.path.join(output_dir, "initial_conditions.vtk")
    )

    # Update variables
    mass, momentum, energy = euler_update(
        density,
        velocity,
        pressure,
        mass,
        momentum,
        energy,
        id_field,
        gamma,
        0.0,
    )

    # Run the simulation
    current_time = 0.0
    save_index = 0
    nr_iterations = 0
    tic = time.time()
    with tqdm(total=nondim_solve_time, desc="Simulation Progress") as pbar:
        while current_time < nondim_solve_time:

            # Get primitive variables
            density, velocity, pressure = conservative_to_primitive(
                density,
                velocity,
                pressure,
                mass,
                momentum,
                energy,
                gamma
            )

            # Get the time step
            dt = get_time_step(
                density,
                velocity,
                pressure,
                id_field,
                courant_factor,
                gamma,
            )

            # Update variables
            mass, momentum, energy = euler_update(
                density,
                velocity,
                pressure,
                mass,
                momentum,
                energy,
                id_field,
                gamma,
                dt
            )

            # Inject fuel
            mass, energy = inject_fuel(
                mass,
                energy,
                id_field,
                mass_flow_rate=0.1,
                energy_flow_rate=1.0,
                center=(0.0, 0.0, 0.0),
                radius=0.0875,
                dt=dt
            )

            # Check if time passes save frequency
            remander = current_time % nondim_save_frequency
            if (remander + dt) > nondim_save_frequency:

                # Save the field
                field_saver(
                    {
                        "density": density,
                        "velocity": velocity,
                        "pressure": pressure,
                        "id_field": id_field
                    },
                    os.path.join(output_dir, f"t_{str(save_index).zfill(4)}.vtk")
                )
                save_index += 1
                print(f"Saved {save_index} files")

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
