# Implosion of a pumpkin cavity

import os
import numpy as np
import warp as wp
from tqdm import tqdm

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
        background_density: wp.float32,
        background_velocity: wp.vec3,
        background_pressure: wp.float32,
        background_magnetic_field: wp.vec3,
        sphere_pressure: wp.float32,
        radius: wp.float32,
    ):

        # Get index
        i, j, k = wp.tid()

        # Get x, y, z
        x = density.origin[0] + wp.float32(i) * density.spacing[0] + 0.5 * density.spacing[0]
        y = density.origin[1] + wp.float32(j) * density.spacing[1] + 0.5 * density.spacing[1]
        z = density.origin[2] + wp.float32(k) * density.spacing[2] + 0.5 * density.spacing[2]

        # Initialize the density, velocity, and pressure
        density.data[0, i, j, k] = background_density
        velocity.data[0, i, j, k] = background_velocity[0]
        velocity.data[1, i, j, k] = background_velocity[1]
        velocity.data[2, i, j, k] = background_velocity[2]
        face_magnetic_field.data[0, i, j, k] = background_magnetic_field[0]
        face_magnetic_field.data[1, i, j, k] = background_magnetic_field[1]
        face_magnetic_field.data[2, i, j, k] = background_magnetic_field[2]
        if x ** 2.0 + y ** 2.0 + z ** 2.0 < radius ** 2.0:
            pressure.data[0, i, j, k] = sphere_pressure + 0.5 * (background_magnetic_field[0] ** 2.0 + background_magnetic_field[1] ** 2.0 + background_magnetic_field[2] ** 2.0)
        else:
            pressure.data[0, i, j, k] = background_pressure + 0.5 * (background_magnetic_field[0] ** 2.0 + background_magnetic_field[1] ** 2.0 + background_magnetic_field[2] ** 2.0)

    def __call__(
        self,
        density,
        velocity,
        pressure,
        face_magnetic_field,
        background_density,
        background_velocity,
        background_pressure,
        background_magnetic_field,
        sphere_pressure,
        radius,
    ):
        # Launch kernel
        wp.launch(
            self._initialize_primitives,
            inputs=[
                density,
                velocity,
                pressure,
                face_magnetic_field,
                background_density,
                background_velocity,
                background_pressure,
                background_magnetic_field,
                sphere_pressure,
                radius,
            ],
            dim=density.shape,
        )

        return density, velocity, pressure, face_magnetic_field


if __name__ == '__main__':

    # Define simulation parameters
    dx = 0.01
    origin = (-1.0, -1.0, -1.0)
    spacing = (dx, dx, dx)
    shape = (int(2.0 / dx), int(2.0 / dx), int(2.0 / dx))
    simulation_time = 1.0
    save_frequency = 0.01
    gamma = (5.0 / 3.0)
    courant_factor = 0.3

    # Initial conditions
    background_density = 1.0
    background_velocity = (0.0, 0.0, 0.0)
    background_pressure = 0.1
    background_magnetic_field = (0.0, 1.0, 0.0)
    sphere_pressure = 10.0
    radius = 0.1

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
        background_density,
        background_velocity,
        background_pressure,
        background_magnetic_field,
        sphere_pressure,
        radius
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

            # Check if NaN
            if (
                np.isnan(density.data.numpy()).any()
                or np.isnan(velocity.data.numpy()).any()
                or np.isnan(pressure.data.numpy()).any()
                or np.isnan(cell_magnetic_field.data.numpy()).any()
            ):
                raise ValueError("NaN detected in the simulation")

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
                        "b": face_magnetic_field,
                        "mass": mass,
                        "mom": momentum,
                        "energy": energy,
                        "flux_b": flux_magnetic_field,
                    },
                    os.path.join(output_dir, f"tt_{str(save_index).zfill(4)}.vtk")
                )
 
            # Update the time
            current_time += dt

            # Update the progress bar
            pbar.update(dt)
