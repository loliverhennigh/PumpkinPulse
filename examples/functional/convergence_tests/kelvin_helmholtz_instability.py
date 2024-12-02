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

from convergence import convergence_analysis

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
        w0 = 0.05
        sigma = 0.05 / wp.sqrt(2.0)
        rho = 1.0 + wp.exp(-(y - 0.25)**2.0 / (2.0 * sigma**2.0)) + wp.exp(-(y - 0.75)**2.0 / (2.0 * sigma**2.0))
        vx = 0.5 * (1.0 - wp.exp(-(y - 0.25)**2.0 / (2.0 * sigma**2.0)) - wp.exp(-(y - 0.75)**2.0 / (2.0 * sigma**2.0)))
        #if (y > 0.25) and (y < 0.75):
        #    rho = 2.0
        #    vx = 0.5
        #else:
        #    rho = 1.0
        #    vx = -0.5
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

def run_sim(factor):

    # Define simulation parameters
    dx = 1.0 / 128.0 / factor
    origin = (0.0, 0.0, 0.0)
    spacing = (dx, dx, 1.0)
    shape = (int(1.0 / dx), int(1.0 / dx), 1)
    print(shape)
    nr_cells = shape[0] * shape[1]
    simulation_time = 2.0
    gamma = 5.0 / 3.0
    courant_factor = 0.4
    dt = 0.001 / factor

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
        spacing=spacing,
        ordering=0,
    )
    velocity = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing,
        ordering=0,
    )
    pressure = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing,
        ordering=0,
    )
    mass = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing,
        ordering=0,
    )
    momentum = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing,
        ordering=0,
    )
    energy = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing,
        ordering=0,
    )
    id_field = constructor.create_field(
        dtype=wp.uint8,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing,
        ordering=0,
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
    total_energy = []
    total_entropy = []
    nr_iterations = 2 * 1024 * factor
    for i in tqdm(range(nr_iterations)):

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
            id_field,
            gamma,
            dt
        )

        # Store the total energy and entropy
        if i % factor == 0:
            total_energy.append(
                np.sum(energy.data.numpy()[..., 0])
            )
            np_density = density.data.numpy()[..., 0]
            np_pressure = pressure.data.numpy()[..., 0]
            cp = 1.0 / (gamma - 1.0)
            total_entropy.append(
                cp * np.sum(np.log(np_pressure / np_density**gamma))
            )

    # Plot the total energy
    plt.close()
    plt.plot(total_energy)
    plt.xlabel("Iteration")
    plt.ylabel("Total Energy")
    plt.title("Total Energy vs. Iteration")
    plt.savefig(os.path.join(output_dir, f"total_energy_{factor}.png"))
    plt.close()
    plt.plot(total_entropy)
    plt.xlabel("Iteration")
    plt.ylabel("Total Entropy")
    plt.title("Total Entropy vs. Iteration")
    plt.savefig(os.path.join(output_dir, f"total_entropy_{factor}.png"))
    plt.close()


    # Save the final state
    field_saver(
        {
            "density": density,
            "velocity": velocity,
            "pressure": pressure,
            "mass": mass,
            "momentum": momentum,
            "energy": energy,
        },
        os.path.join(output_dir, f"final_state_{factor}.vtk")
    )

    # Return the final state
    wp.synchronize()
    yield {
        "density": density.data.numpy()[..., 0],
        "velocity": velocity.data.numpy()[..., 0],
        "pressure": pressure.data.numpy()[..., 0],
    }
    wp.synchronize()

if __name__ == "__main__":
    # Run convergence analysis
    output_dir = "output"
    convergence_analysis(
        run_sim,
        factors=[2 ** i for i in range(4)],
        output_dir=output_dir,
    )
