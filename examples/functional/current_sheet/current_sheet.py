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
from pumpkin_pulse.operator.mhd import (
    PrimitiveToConservative,
    ConservativeToPrimitive,
    GetTimeStep,
    IdealMHDUpdate,
    ConstrainedTransport,
    FaceMagneticFieldToCellMagneticField,
)
from pumpkin_pulse.operator.saver import FieldSaver

class InitializePrimitives(Operator):

    @wp.kernel
    def _initialize_primitives(
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
        face_magnetic_field: Fieldfloat32,
    ):

        # Get index
        i, j, k = wp.tid()

        # Get x, y, z
        x = density.origin[0] + wp.float32(i) * density.spacing[0] + 0.5 * density.spacing[0]
        y = density.origin[1] + wp.float32(j) * density.spacing[1] + 0.5 * density.spacing[1]
        z = density.origin[2] + wp.float32(k) * density.spacing[2] + 0.5 * density.spacing[2]

        # Initialize the primitives
        density.data[0, i, j, k] = 1.0
        velocity.data[0, i, j, k] = 0.1 * wp.sin(2.0 * 3.14159265359 * y)
        velocity.data[1, i, j, k] = 0.0
        velocity.data[2, i, j, k] = 0.0
        pressure.data[0, i, j, k] = 0.1 + 0.5
        if x < 0.5:
            face_magnetic_field.data[0, i, j, k] = 0.0
            face_magnetic_field.data[1, i, j, k] = 1.0
            face_magnetic_field.data[2, i, j, k] = 0.0
        elif x >= 0.5 and x < 1.5:
            face_magnetic_field.data[0, i, j, k] = 0.0
            face_magnetic_field.data[1, i, j, k] = -1.0
            face_magnetic_field.data[2, i, j, k] = 0.0
        else:
            face_magnetic_field.data[0, i, j, k] = 0.0
            face_magnetic_field.data[1, i, j, k] = 1.0
            face_magnetic_field.data[2, i, j, k] = 0.0

    def __call__(
        self,
        density,
        velocity,
        pressure,
        face_magnetic_field,
    ):
        # Launch kernel
        wp.launch(
            self._initialize_primitives,
            inputs=[
                density,
                velocity,
                pressure,
                face_magnetic_field,
            ],
            dim=density.shape,
        )

        return density, velocity, pressure, face_magnetic_field


if __name__ == '__main__':

    # Define simulation parameters
    dx = 0.0005
    origin = (0.0, 0.0, 0.0)
    spacing = (dx, dx, 1.0)
    shape = (int(2.0 / dx), int(2.0 / dx), 1)
    nr_cells = shape[0] * shape[1] * shape[2]
    simulation_time = 25.0
    save_frequency = 0.01
    gamma = (5.0 / 3.0)
    courant_factor = 0.3

    # Make output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Make the constructor
    constructor = Constructor()

    # Make the operators
    initialize_primitives = InitializePrimitives()
    primitive_to_conservative = PrimitiveToConservative()
    conservative_to_primitive = ConservativeToPrimitive()
    get_time_step = GetTimeStep()
    ideal_mhd_update = IdealMHDUpdate()
    constrained_transport = ConstrainedTransport()
    face_magnetic_field_to_cell_magnetic_field = FaceMagneticFieldToCellMagneticField()
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
    cell_magnetic_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
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
    flux_magnetic_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=6,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    face_magnetic_field = constructor.create_field(
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

    # Initialize the cavity
    density, velocity, pressure, face_magnetic_field = initialize_primitives(
        density,
        velocity,
        pressure,
        face_magnetic_field,
    )

    # Get cell magnetic field
    cell_magnetic_field = face_magnetic_field_to_cell_magnetic_field(
        face_magnetic_field,
        cell_magnetic_field,
    )

    # Get conservative variables
    mass, momentum, energy = primitive_to_conservative(
        density,
        velocity,
        pressure,
        cell_magnetic_field,
        mass,
        momentum,
        energy,
        gamma
    )

    # Save the fields
    field_saver(
        {"density": density, "velocity": velocity, "pressure": pressure, "cell_magnetic_field": cell_magnetic_field},
        os.path.join(output_dir, "initial_conditions.vtk")
    )
    field_saver(
        {"mass": mass, "momentum": momentum, "energy": energy, "face_magnetic_field": face_magnetic_field},
        os.path.join(output_dir, "initial_conserved_variables.vtk")
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
                cell_magnetic_field,
                mass,
                momentum,
                energy,
                gamma
            )

            ## Check if NaN
            #if (
            #    np.isnan(density.data.numpy()).any()
            #    or np.isnan(velocity.data.numpy()).any()
            #    or np.isnan(pressure.data.numpy()).any()
            #    or np.isnan(cell_magnetic_field.data.numpy()).any()
            #):
            #    raise ValueError("NaN detected in the simulation")

            # Get the time step
            dt = get_time_step(
                density,
                velocity,
                pressure,
                cell_magnetic_field,
                id_field,
                courant_factor,
                gamma,
            )

            # Update Conserved Variables
            mass, momentum, energy, flux_magnetic_field = ideal_mhd_update(
                density,
                velocity,
                pressure,
                cell_magnetic_field,
                mass,
                momentum,
                energy,
                flux_magnetic_field,
                id_field,
                gamma,
                dt
            )

            # Update the magnetic field
            face_magnetic_field = constrained_transport(
                face_magnetic_field,
                flux_magnetic_field,
                dt
            )

            # Update the cell magnetic field
            cell_magnetic_field = face_magnetic_field_to_cell_magnetic_field(
                face_magnetic_field,
                cell_magnetic_field,
            )

            # Check if time passes save frequency
            remander = current_time % save_frequency 
            if (remander + dt) > save_frequency:
                save_index += 1
                print(f"Saved {save_index} files")
                print(dt)
                field_saver(
                    {
                        "density": density,
                        "velocity": velocity,
                        "pressure": pressure,
                        "b": face_magnetic_field,
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
