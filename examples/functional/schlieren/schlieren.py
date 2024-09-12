# Implosion of a pumpkin cavity

import os
import numpy as np
import warp as wp
from build123d import Rectangle, extrude, Sphere, Location, Circle, Rotation
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

####### STL file ########
# Pumpkin Patch by tone001 on Thingiverse: https://www.thingiverse.com/thing:1836056
# This thing was created by Thingiverse user tone001, and is licensed under Creative Commons - Attribution - Non-Commercial - Share Alike
#########################

class CavityInitialize(Operator):

    @wp.kernel
    def _initialize_cavity(
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
        id_field: Fielduint8,
    ):

        # Get index
        i, j, k = wp.tid()

        # Get the id
        id = id_field.data[0, i, j, k]

        # Get x, y, z position
        x = density.origin[0] + wp.float32(i) * density.spacing[0]
        y = density.origin[1] + wp.float32(j) * density.spacing[1]
        z = density.origin[2] + wp.float32(k) * density.spacing[2]

        # Gaussian source wave
        if x ** 2.0 + y ** 2.0 + z ** 2.0 > 0.3**2.0:
            pressure.data[0, i, j, k] = 100.0
            density.data[0, i, j, k] = 1.0
        elif wp.int32(id) == 1:
            pressure.data[0, i, j, k] = 0.1
            density.data[0, i, j, k] = 0.1
        else:
            pressure.data[0, i, j, k] = 1.0
            density.data[0, i, j, k] = 1.0

        # Initialize the velocity
        velocity.data[0, i, j, k] = 0.0
        velocity.data[1, i, j, k] = 0.0
        velocity.data[2, i, j, k] = 0.0

    def __call__(
        self,
        density,
        velocity,
        pressure,
        id_field,
    ):
        # Launch kernel
        wp.launch(
            self._initialize_cavity,
            inputs=[
                density,    
                velocity,
                pressure,
                id_field,
            ],
            dim=density.shape,
        )

        return density, velocity, pressure
 
if __name__ == '__main__':

    # Define simulation parameters
    dx = 0.0015
    origin = (-0.4, -0.4, -0.4)
    spacing = (dx, dx, dx)
    shape = (int(0.8 / dx), int(0.8 / dx), int(0.8 / dx))
    nr_cells = shape[0] * shape[1] * shape[2]
    simulation_time = 0.1
    gamma = (5.0 / 3.0)
    courant_factor = 0.2

    # Make output directory
    save_frequency = 0.001
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Make the constructor
    constructor = Constructor()

    # Make the operators
    cavity_initialize = CavityInitialize()
    primitive_to_conservative = PrimitiveToConservative()
    conservative_to_primitive = ConservativeToPrimitive()
    get_time_step = GetTimeStep()
    euler_update = EulerUpdate()
    field_saver = FieldSaver()
    stl_to_mesh = StlToMesh()
    mesh_to_id_field = MeshToIdField()

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

    # Make mesh
    mesh = stl_to_mesh("files/Pumpkin_whole.stl")
    id_field = mesh_to_id_field(mesh, id_field, 1)

    # Save id field
    field_saver({"id_field": id_field}, os.path.join(output_dir, "id_field.vtk"))

    # Initialize the cavity
    density, velocity, pressure = cavity_initialize(density, velocity, pressure, id_field)

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

    # Zero id field
    id_field.data.zero_()

    # Save the fields
    field_saver(
        {"density": density, "velocity": velocity, "pressure": pressure},
        os.path.join(output_dir, "initial_conditions.vtk")
    )
    field_saver(
        {"mass": mass, "momentum": momentum, "energy": energy},
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

            ## Print total mass
            #total_mass = np.sum(mass.data.numpy())
            #print(f"Total mass: {total_mass}")

            # Check if time passes save frequency
            remander = current_time % save_frequency 
            if (remander + dt) > save_frequency:
                field_saver(
                    {
                        "density": density,
                        "velocity": velocity,
                        "pressure": pressure,
                        #"mass": mass,
                        #"momentum": momentum,
                        #"energy": energy,
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
