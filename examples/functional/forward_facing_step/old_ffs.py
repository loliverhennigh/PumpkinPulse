# Forward facing step simulation using the pumpkin_pulse library

import os
import numpy as np
import warp as wp
from tqdm import tqdm

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

# Global parameters
density_inlet = 1.4
velocity_inlet = 3.0
pressure_inlet = 1.0
step_height = 0.2
start_x = 0.6

@wp.func
def inside_cavity(
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
    origin: wp.vec3f,
    spacing: wp.vec3f,
    shape: wp.vec3i
):
    # Get the position
    x = origin[0] + wp.float32(i) * spacing[0]
    y = origin[1] + wp.float32(j) * spacing[1]
    z = origin[2] + wp.float32(k) * spacing[2]

    # Check if top or bottom
    if j == 0 or j == shape[1] - 1:
        return wp.uint8(0)
    elif i == 0 or i == shape[0] - 1:
        return wp.uint8(0)
    # circle
    elif (x - 0.5) ** 2.0 + (y - 0.5) ** 2.0 < 0.1 ** 2.0:
        return wp.uint8(0)
    else:
        return wp.uint8(1)


class InitializeVel(Operator):

    @wp.kernel
    def _initialize_vel(
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
    ):

        # Get index
        i, j, k = wp.tid()

        # Initialize the density, velocity, and pressure
        density.data[0, i, j, k] = density_inlet
        velocity.data[0, i, j, k] = velocity_inlet
        velocity.data[1, i, j, k] = 0.0
        velocity.data[2, i, j, k] = 0.0
        pressure.data[0, i, j, k] = pressure_inlet

    def __call__(
        self,
        density,
        velocity,
        pressure,
    ):
        # Launch kernel
        wp.launch(
            self._initialize_vel,
            inputs=[
                density,    
                velocity,
                pressure,
            ],
            dim=density.shape,
        )

        return density, velocity, pressure


@wp.func
def apply_ffs_boundary_conditions(
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
    origin: wp.vec3f,
    spacing: wp.vec3f,
    shape: wp.vec3i,
    rho_faces: EulerUpdate._scalar_faces_type,
    vx_faces: EulerUpdate._scalar_faces_type,
    vy_faces: EulerUpdate._scalar_faces_type,
    vz_faces: EulerUpdate._scalar_faces_type,
    p_faces: EulerUpdate._scalar_faces_type,
    gamma: wp.float32,
    dt: wp.float32,
):

    # Check if inlet cell
    if i == 1:
        # Set dirichlet boundary conditions
        rho_stencil[1] = density_inlet
        vx_stencil[1] = velocity_inlet
        vy_stencil[1] = 0.0
        vz_stencil[1] = 0.0
        p_stencil[1] = pressure_inlet

        # Set the derivatives
        for d in range(3):
            rho_stencil_dxyz[1, d] = 0.0
            vx_stencil_dxyz[1, d] = 0.0
            vy_stencil_dxyz[1, d] = 0.0
            vz_stencil_dxyz[1, d] = 0.0
            p_stencil_dxyz[1, d] = 0.0

    # Check if outlet cell
    if i == shape[0] - 2:
        # Set dirichlet boundary conditions
        rho_stencil[2] = rho_stencil[0]
        vx_stencil[2] = vx_stencil[0]
        vy_stencil[2] = vy_stencil[0]
        vz_stencil[2] = vz_stencil[0]
        p_stencil[2] = p_stencil[0]

        # Set the derivatives
        for d in range(3):
            rho_stencil_dxyz[2, d] = 0.0
            vx_stencil_dxyz[2, d] = 0.0
            vy_stencil_dxyz[2, d] = 0.0
            vz_stencil_dxyz[2, d] = 0.0
            p_stencil_dxyz[2, d] = 0.0

    # Check for no slip boundary conditions
    for d in range(3):
        for s in range(2):

            # Get offset
            if d == 0:
                flip_x = -1.0
                i_offset = -1 + 2 * s
            else:
                flip_x = 1.0
                i_offset = 0
            if d == 1:
                flip_y = -1.0
                j_offset = -1 + 2 * s
            else:
                flip_y = 1.0
                j_offset = 0
            if d == 2:
                flip_z = -1.0
                k_offset = -1 + 2 * s
            else:
                flip_z = 1.0
                k_offset = 0
            
            # Check if any side is outside the cavity
            is_inside = inside_cavity(
                i + i_offset,
                j + j_offset,
                k + k_offset,
                origin,
                spacing,
                shape
            )

            # Apply slip boundary conditions
            if (is_inside == wp.uint8(0)) and (i + i_offset != 0) and (i + i_offset != shape[0] - 1):
                # Set dirichlet boundary conditions
                rho_stencil[2 * d + s + 1] = rho_stencil[0]
                vx_stencil[2 * d + s + 1] = flip_x * vx_stencil[0]
                vy_stencil[2 * d + s + 1] = flip_y * vy_stencil[0]
                vz_stencil[2 * d + s + 1] = flip_z * vz_stencil[0]
                p_stencil[2 * d + s + 1] = p_stencil[0]

                # Set the derivatives
                rho_stencil_dxyz[2 * d + s + 1, 0] = rho_stencil_dxyz[0, 0]
                rho_stencil_dxyz[2 * d + s + 1, 1] = rho_stencil_dxyz[0, 1]
                rho_stencil_dxyz[2 * d + s + 1, 2] = rho_stencil_dxyz[0, 2]
                vx_stencil_dxyz[2 * d + s + 1, 0] = vx_stencil_dxyz[0, 0]
                vx_stencil_dxyz[2 * d + s + 1, 1] = vx_stencil_dxyz[0, 1]
                vx_stencil_dxyz[2 * d + s + 1, 2] = vx_stencil_dxyz[0, 2]
                vy_stencil_dxyz[2 * d + s + 1, 0] = vy_stencil_dxyz[0, 0]
                vy_stencil_dxyz[2 * d + s + 1, 1] = vy_stencil_dxyz[0, 1]
                vy_stencil_dxyz[2 * d + s + 1, 2] = vy_stencil_dxyz[0, 2]
                vz_stencil_dxyz[2 * d + s + 1, 0] = vz_stencil_dxyz[0, 0]
                vz_stencil_dxyz[2 * d + s + 1, 1] = vz_stencil_dxyz[0, 1]
                vz_stencil_dxyz[2 * d + s + 1, 2] = vz_stencil_dxyz[0, 2]
                p_stencil_dxyz[2 * d + s + 1, 0] = p_stencil_dxyz[0, 0]
                p_stencil_dxyz[2 * d + s + 1, 1] = p_stencil_dxyz[0, 1]
                p_stencil_dxyz[2 * d + s + 1, 2] = p_stencil_dxyz[0, 2]
                #rho_stencil_dxyz[2 * d + s + 1, 0] = flip_x * rho_stencil_dxyz[0, 0]
                #rho_stencil_dxyz[2 * d + s + 1, 1] = flip_y * rho_stencil_dxyz[0, 1]
                #rho_stencil_dxyz[2 * d + s + 1, 2] = flip_z * rho_stencil_dxyz[0, 2]
                #vx_stencil_dxyz[2 * d + s + 1, 0] = flip_x * vx_stencil_dxyz[0, 0]
                #vx_stencil_dxyz[2 * d + s + 1, 1] = flip_y * vx_stencil_dxyz[0, 1]
                #vx_stencil_dxyz[2 * d + s + 1, 2] = flip_z * vx_stencil_dxyz[0, 2]
                #vy_stencil_dxyz[2 * d + s + 1, 0] = flip_x * vy_stencil_dxyz[0, 0]
                #vy_stencil_dxyz[2 * d + s + 1, 1] = flip_y * vy_stencil_dxyz[0, 1]
                #vy_stencil_dxyz[2 * d + s + 1, 2] = flip_z * vy_stencil_dxyz[0, 2]
                #vz_stencil_dxyz[2 * d + s + 1, 0] = flip_x * vz_stencil_dxyz[0, 0]
                #vz_stencil_dxyz[2 * d + s + 1, 1] = flip_y * vz_stencil_dxyz[0, 1]
                #vz_stencil_dxyz[2 * d + s + 1, 2] = flip_z * vz_stencil_dxyz[0, 2]
                #p_stencil_dxyz[2 * d + s + 1, 0] = flip_x * p_stencil_dxyz[0, 0]
                #p_stencil_dxyz[2 * d + s + 1, 1] = flip_y * p_stencil_dxyz[0, 1]
                #p_stencil_dxyz[2 * d + s + 1, 2] = flip_z * p_stencil_dxyz[0, 2]


    return rho_stencil, rho_stencil_dxyz, vx_stencil, vx_stencil_dxyz, vy_stencil, vy_stencil_dxyz, vz_stencil, vz_stencil_dxyz, p_stencil, p_stencil_dxyz




if __name__ == '__main__':

    # Define simulation parameters
    dx = 0.002
    origin = (0.0, 0.0, 0.0)
    spacing = (dx, dx, dx)
    shape = (int(3.0 / dx), int(1.0 / dx), 1)
    simulation_time = 2.5
    gamma = (5.0 / 3.0)
    courant_factor = 0.2

    # Make output directory
    save_frequency = 0.01
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Make the constructor
    constructor = Constructor()

    # Make the operators
    initialize_vel = InitializeVel()
    primitive_to_conservative = PrimitiveToConservative()
    conservative_to_primitive = ConservativeToPrimitive()
    get_time_step = GetTimeStep()
    euler_update = EulerUpdate(
        inside_domain=inside_cavity,
        apply_boundary_conditions=apply_ffs_boundary_conditions)
    field_saver = FieldSaver()

    # Make the fields
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

    # Initialize the cavity
    density, velocity, pressure = initialize_vel(
        density,
        velocity,
        pressure
    )

    # Get conservative variables
    mass, momentum, energy = primitive_to_conservative(
        density,
        velocity,
        pressure,
        mass,
        momentum,
        energy,
        gamma
    )

    # Run the simulation
    current_time = 0.0
    save_index = 0
    with tqdm(total=simulation_time, desc="Simulation Progress") as pbar:
        while current_time < simulation_time:

            # Get the time step
            dt = get_time_step(
                density,
                velocity,
                pressure,
                courant_factor,
                gamma,
            )

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

            # Update variables
            mass, momentum, energy = euler_update(
                density,
                velocity,
                pressure,
                mass,
                momentum,
                energy,
                gamma,
                dt
            )


            # Check if time passes save frequency
            remander = current_time % save_frequency 
            if (remander + dt) > save_frequency:
                field_saver(
                    {"density": density, "velocity": velocity, "pressure": pressure},
                    os.path.join(output_dir, f"t_{str(save_index).zfill(4)}.vtk")
                )
                save_index += 1
                print(np.sum(mass.data.numpy()))
                print(f"Saved {save_index} files")

            # Update the time
            current_time += dt

            # Update the progress bar
            pbar.update(dt)
