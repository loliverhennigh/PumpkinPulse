# Rayleigh instability simulation using the pumpkin_pulse library

import os
import numpy as np
import warp as wp
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    #if j == 0 or j == shape[1] - 1:
    #    return wp.uint8(0)
    #if ((x - 0.5)**2.0 + (y - 0.5)**2.0) < 0.1**2.0:
    #    return wp.uint8(0)
    if (x > 0.4) and (x < 0.6) and (y > 0.4) and (y < 0.6):
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

        # Get x, y location
        x = density.origin[0] + wp.float32(i) * density.spacing[0]
        y = density.origin[1] + wp.float32(j) * density.spacing[1]

        # Initialize the density, velocity, and pressure
        #w0 = 0.3
        #sigma = 0.05 / wp.sqrt(2.0)
        rho = 1.0
        vx = 3.0
        vy = 0.0
        p = 1.3
        #if (y > 0.25) and (y < 0.5):
        #    rho = 2.0
        #    vx = 0.5
        #    #vy = 0.1
        #    vy = w0 * wp.sin(4.0 * 3.14159 * x) * (wp.exp(-(y - 0.25)**2.0 / (2.0 * sigma**2.0)) + wp.exp(-(y - 0.75)**2.0 / (2.0 * sigma**2.0))) + 0.01
        #else:
        #    rho = 1.0
        #    vx = -0.5
        #    vy = 0.0

        # Set the values
        density.data[0, i, j, k] = rho
        velocity.data[0, i, j, k] = vx
        velocity.data[1, i, j, k] = vy
        velocity.data[2, i, j, k] = 0.0
        pressure.data[0, i, j, k] = p

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
    rho_stencil: EulerUpdate._scalar_stencil_type,
    rho_stencil_dxyz: EulerUpdate._vector_stencil_type,
    vx_stencil: EulerUpdate._scalar_stencil_type,
    vx_stencil_dxyz: EulerUpdate._vector_stencil_type,
    vy_stencil: EulerUpdate._scalar_stencil_type,
    vy_stencil_dxyz: EulerUpdate._vector_stencil_type,
    vz_stencil: EulerUpdate._scalar_stencil_type,
    vz_stencil_dxyz: EulerUpdate._vector_stencil_type,
    p_stencil: EulerUpdate._scalar_stencil_type,
    p_stencil_dxyz: EulerUpdate._vector_stencil_type,
    gamma: wp.float32,
    dt: wp.float32,
):
    ## Check if inlet cell
    #if j == 1:
    #    # Set dirichlet boundary conditions
    #    rho_stencil[3] = rho_stencil[0]
    #    vx_stencil[3] = vx_stencil[0]
    #    vy_stencil[3] = -vy_stencil[0]
    #    vz_stencil[3] = vz_stencil[0]
    #    p_stencil[3] = p_stencil[0]

    #    # Set the derivatives
    #    rho_stencil_dxyz[3, 0] = rho_stencil_dxyz[0, 0]
    #    vx_stencil_dxyz[3, 0] = vx_stencil_dxyz[0, 0]
    #    vy_stencil_dxyz[3, 0] = vy_stencil_dxyz[0, 0]
    #    vz_stencil_dxyz[3, 0] = vz_stencil_dxyz[0, 0]
    #    p_stencil_dxyz[3, 0] = p_stencil_dxyz[0, 0]
    #    rho_stencil_dxyz[3, 1] = -rho_stencil_dxyz[0, 1]
    #    vx_stencil_dxyz[3, 1] = -vx_stencil_dxyz[0, 1]
    #    vy_stencil_dxyz[3, 1] = -vy_stencil_dxyz[0, 1]
    #    vz_stencil_dxyz[3, 1] = -vz_stencil_dxyz[0, 1]
    #    p_stencil_dxyz[3, 1] = -p_stencil_dxyz[0, 1]
    #    rho_stencil_dxyz[3, 2] = rho_stencil_dxyz[0, 2]
    #    vx_stencil_dxyz[3, 2] = vx_stencil_dxyz[0, 2]
    #    vy_stencil_dxyz[3, 2] = vy_stencil_dxyz[0, 2]
    #    vz_stencil_dxyz[3, 2] = vz_stencil_dxyz[0, 2]
    #    p_stencil_dxyz[3, 2] = p_stencil_dxyz[0, 2]


    ## Check if outlet cell
    #if j == shape[1] - 2:
    #    # Set dirichlet boundary conditions
    #    rho_stencil[4] = rho_stencil[0]
    #    vx_stencil[4] = vx_stencil[0]
    #    vy_stencil[4] = -vy_stencil[0]
    #    vz_stencil[4] = vz_stencil[0]
    #    p_stencil[4] = p_stencil[0]

    #    # Set the derivatives
    #    rho_stencil_dxyz[4, 0] = rho_stencil_dxyz[0, 0]
    #    vx_stencil_dxyz[4, 0] = vx_stencil_dxyz[0, 0]
    #    vy_stencil_dxyz[4, 0] = vy_stencil_dxyz[0, 0]
    #    vz_stencil_dxyz[4, 0] = vz_stencil_dxyz[0, 0]
    #    p_stencil_dxyz[4, 0] = p_stencil_dxyz[0, 0]
    #    rho_stencil_dxyz[4, 1] = -rho_stencil_dxyz[0, 1]
    #    vx_stencil_dxyz[4, 1] = -vx_stencil_dxyz[0, 1]
    #    vy_stencil_dxyz[4, 1] = -vy_stencil_dxyz[0, 1]
    #    vz_stencil_dxyz[4, 1] = -vz_stencil_dxyz[0, 1]
    #    p_stencil_dxyz[4, 1] = -p_stencil_dxyz[0, 1]
    #    rho_stencil_dxyz[4, 2] = rho_stencil_dxyz[0, 2]
    #    vx_stencil_dxyz[4, 2] = vx_stencil_dxyz[0, 2]
    #    vy_stencil_dxyz[4, 2] = vy_stencil_dxyz[0, 2]
    #    vz_stencil_dxyz[4, 2] = vz_stencil_dxyz[0, 2]
    #    p_stencil_dxyz[4, 2] = p_stencil_dxyz[0, 2]


    # Check for no slip boundary conditions
    for d in range(3):
        for s in range(2):

            # Get offset
            if d == 0:
                i_offset = -1 + 2 * s
                flip_x = -1.0
            else:
                i_offset = 0
                flip_x = 1.0
            if d == 1:
                j_offset = -1 + 2 * s
                flip_y = -1.0
            else:
                j_offset = 0
                flip_y = 1.0
            if d == 2:
                k_offset = -1 + 2 * s
                flip_z = -1.0
            else:
                k_offset = 0
                flip_z = 1.0

            
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
            #if (is_inside == wp.uint8(0)) and (j + j_offset != 0) and (j + j_offset != shape[0] - 1):
            if (is_inside == wp.uint8(0)):
                # boundary condition
                if d == 0:

                    # dirichlet boundary conditions
                    rho_stencil[2 * d + s + 1] = rho_stencil[0]
                    vx_stencil[2 * d + s + 1] = -vx_stencil[0]
                    vy_stencil[2 * d + s + 1] = vy_stencil[0]
                    p_stencil[2 * d + s + 1] = p_stencil[0]

                    # derivatives
                    rho_stencil_dxyz[2 * d + s + 1, 0] = -rho_stencil_dxyz[0, 0]
                    vx_stencil_dxyz[2 * d + s + 1, 0] = vx_stencil_dxyz[0, 0]
                    vy_stencil_dxyz[2 * d + s + 1, 0] = vy_stencil_dxyz[0, 0]
                    p_stencil_dxyz[2 * d + s + 1, 0] = -p_stencil_dxyz[0, 0]

                    #rho_stencil_dxyz[2 * d + s + 1, 1] = rho_stencil_dxyz[0, 1]
                    #vx_stencil_dxyz[2 * d + s + 1, 1] = vx_stencil_dxyz[0, 1]
                    #vy_stencil_dxyz[2 * d + s + 1, 1] = vy_stencil_dxyz[0, 1]
                    #p_stencil_dxyz[2 * d + s + 1, 1] = p_stencil_dxyz[0, 1]

                if d == 1:

                    # dirichlet boundary conditions
                    rho_stencil[2 * d + s + 1] = rho_stencil[0]
                    vx_stencil[2 * d + s + 1] = vx_stencil[0]
                    vy_stencil[2 * d + s + 1] = -vy_stencil[0]
                    p_stencil[2 * d + s + 1] = p_stencil[0]

                    # derivatives
                    #rho_stencil_dxyz[2 * d + s + 1, 0] = rho_stencil_dxyz[0, 0]
                    #vx_stencil_dxyz[2 * d + s + 1, 0] = vx_stencil_dxyz[0, 0]
                    #vy_stencil_dxyz[2 * d + s + 1, 0] = vy_stencil_dxyz[0, 0]
                    #p_stencil_dxyz[2 * d + s + 1, 0] = p_stencil_dxyz[0, 0]

                    rho_stencil_dxyz[2 * d + s + 1, 1] = -rho_stencil_dxyz[0, 1]
                    vx_stencil_dxyz[2 * d + s + 1, 1] = vx_stencil_dxyz[0, 1]
                    vy_stencil_dxyz[2 * d + s + 1, 1] = vy_stencil_dxyz[0, 1]
                    p_stencil_dxyz[2 * d + s + 1, 1] = -p_stencil_dxyz[0, 1]






    return rho_stencil, rho_stencil_dxyz, vx_stencil, vx_stencil_dxyz, vy_stencil, vy_stencil_dxyz, vz_stencil, vz_stencil_dxyz, p_stencil, p_stencil_dxyz



if __name__ == '__main__':

    # Define simulation parameters
    dx = 1.0 / 128.0
    origin = (0.0, 0.0, 0.0)
    spacing = (dx, dx, 1.0)
    shape = (int(30.0 / dx), int(3.0 / dx), 1)
    simulation_time = 5.0
    gamma = 1.4

    courant_factor = 0.4

    # Make output directory
    save_frequency = 0.02
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
        inside_cavity,
        apply_ffs_boundary_conditions
    )
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
    total_mass = []
    total_momentum_x = []
    total_momentum_y = []
    total_momentum_z = []
    total_momentum = []
    total_energy = []
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
                total_mass.append(np.sum(mass.data.numpy()))
                total_momentum_x.append(np.sum(momentum.data[0].numpy()))
                total_momentum_y.append(np.sum(momentum.data[1].numpy()))
                total_momentum_z.append(np.sum(momentum.data[2].numpy()))
                total_momentum.append(np.sum(momentum.data.numpy()))
                total_energy.append(np.sum(energy.data.numpy()))
                save_index += 1
                print(f"Saved {save_index} files")

            # Update the time
            current_time += dt

            # Update the progress bar
            pbar.update(dt)

        plt.plot(total_mass)
        plt.title("Total Mass")
        plt.savefig("total_mass.png")
        plt.clf()
        plt.plot(total_energy)
        plt.title("Total Energy")
        plt.savefig("total_energy.png")
        plt.clf()
        plt.plot(total_momentum_z)
        plt.title("Total Momentum Z")
        plt.savefig("total_momentum_z.png")
        plt.clf()
        plt.plot(total_momentum_y)
        plt.title("Total Momentum Y")
        plt.savefig("total_momentum_y.png")
        plt.clf()
        plt.plot(total_momentum_x)
        plt.title("Total Momentum X")
        plt.savefig("total_momentum_x.png")
        plt.clf()
        plt.plot(total_momentum)
        plt.title("Total Momentum")
        plt.savefig("total_momentum.png")
        plt.clf()

 
