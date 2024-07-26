# Kelvin-Helmholtz instability simulation using the pumpkin_pulse library

import os
import numpy as np
import warp as wp
from tqdm import tqdm
import matplotlib.pyplot as plt
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

from finitevolume_python.finitevolume import main

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
        x = density.origin[0] + wp.float32(i) * density.spacing[0] + 0.5 * density.spacing[0]
        y = density.origin[1] + wp.float32(j) * density.spacing[1] + 0.5 * density.spacing[1]

        # Initialize the density, velocity, and pressure
        w0 = 0.1
        sigma = 0.05 / wp.sqrt(2.0)
        if (y > 0.25) and (y < 0.75):
            rho = 2.0
            vx = 0.5
        else:
            rho = 1.0
            vx = -0.5
        vy = w0 * wp.sin(4.0 * 3.14159 * x) * (wp.exp(-(y - 0.25)**2.0 / (2.0 * sigma**2.0)) + wp.exp(-(y - 0.75)**2.0 / (2.0 * sigma**2.0)))
        p = 2.5

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


if __name__ == '__main__':

    # Define simulation parameters
    dx = 0.0001
    origin = (0.0, 0.0, 0.0)
    spacing = (dx, dx, 1.0)
    shape = (int(1.0 / dx), int(1.0 / dx), 1)
    nr_cells = shape[0] * shape[1]
    simulation_time = 2.0
    gamma = 5.0 / 3.0
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
    euler_update = EulerUpdate()
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
    id_field = constructor.create_field(
        dtype=wp.uint8,
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
    nr_iterations = 0
    tic = time.time()
    with tqdm(total=simulation_time, desc="Simulation Progress") as pbar:
        while current_time < simulation_time:

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

            ## Check if time passes save frequency
            ##remander = current_time % save_frequency 
            ##if (remander + dt) > save_frequency:
            ##print("Rho prime")
            ##print(np.min(mass.data.numpy()), np.max(mass.data.numpy()))
            ##print("Vx prime")
            ##print(np.min(momentum.data.numpy()[0]), np.max(momentum.data.numpy()[0]))
            ##print("Vy prime")
            ##print(np.min(momentum.data.numpy()[1]), np.max(momentum.data.numpy()[1]))
            ##print("P prime")
            ##print(np.min(energy.data.numpy()), np.max(energy.data.numpy()))
            #fv_flux_mass_face_x = main()
            #print(dt)

            ## plot comparison
            #fig, axs = plt.subplots(3, 1)
            #axs[0].imshow(momentum.data.numpy()[0, :, :, 0])
            #axs[0].set_title("Density")
            #axs[0].axis("off")
            #axs[1].imshow(fv_flux_mass_face_x)
            #axs[1].set_title("Density Flux X")
            #axs[1].axis("off")
            #axs[2].imshow((momentum.data.numpy()[0, :, :, 0] - fv_flux_mass_face_x))
            #axs[2].set_title("Density - Density Flux X")
            #axs[2].axis("off")
            #plt.show()
            #exit()

            remander = current_time % save_frequency
            if (remander + dt) > save_frequency:
                print("Here")
                field_saver(
                    {
                        "density": density,
                        "velocity": velocity,
                        "pressure": pressure,
                        "mass": mass,
                        "momentum": momentum,
                        "energy": energy,
                    },
                    os.path.join(output_dir, f"t_{str(save_index).zfill(4)}.vtk")
                )
                save_index += 1


            # Compute MUPS
            if nr_iterations % 10 == 0:
                wp.synchronize()
                toc = time.time()
                mups = nr_cells * nr_iterations / (toc - tic) / 1.0e6
                print(f"Elapsed time: {toc - tic}")
                print(f"Iterations: {nr_iterations}")
                print(f"MUPS: {mups}")


            #exit()
            #print(f"Saved {save_index} files")

            # Update the time
            current_time += dt

            # Update the progress bar
            pbar.update(dt)

            # Update the number of iterations
            nr_iterations += 1
