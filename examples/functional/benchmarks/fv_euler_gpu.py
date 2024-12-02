import time
import numpy as np
import warp as wp
from tqdm import tqdm
import argparse

wp.init()

from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.data import Fieldfloat32, Fielduint8
from pumpkin_pulse.operator.hydro.euler_eos import (
    ConservativeToPrimitive,
    TwoKernelEulerUpdate,
)

# Argument parser for resolution
parser = argparse.ArgumentParser(description='Run a simple FV Euler simulation on the GPU.')
parser.add_argument('--resolution', type=int, default=256, help='Resolution of the simulation.')
resolution = parser.parse_args().resolution

if __name__ == '__main__':

    # Define simulation parameters
    dx = 1.0
    origin = (0.0, 0.0, 0.0)
    spacing = (dx, dx, dx)
    shape = (resolution, resolution, resolution)
    nr_cells = shape[0] * shape[1] * shape[2]

    # Electric parameters

    # Use CFL condition to determine time step
    dt = 0.0000001
    num_steps = 256
    print(f"Number of steps: {num_steps}")

    # Make the constructor
    constructor = Constructor()

    # Make the operators
    conservative_to_primitive = ConservativeToPrimitive()
    two_kernel_euler_update = TwoKernelEulerUpdate()

    # Make the fields
    id_field = constructor.create_field(
        dtype=wp.uint8,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing,
        ordering=0,
    )
    conservative_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=5,
        shape=shape,
        origin=origin,
        spacing=spacing,
        ordering=0,
    )
    primitive_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=5,
        shape=shape,
        origin=origin,
        spacing=spacing,
        ordering=0,
    )
    primitive_dxyz_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=15,
        shape=shape,
        origin=origin,
        spacing=spacing,
        ordering=0,
    )

    # Run 1 step for kernel initialization
    #primitive_field = conservative_to_primitive(
    #    primitive_field,
    #    conservative_field,
    #    gamma=1.4,
    #)
    conservative_field = two_kernel_euler_update(
        primitive_field,
        primitive_dxyz_field,
        conservative_field,
        gamma=1.4,
        dt=dt,
    )

    # Run the simulation
    wp.synchronize()
    tic = time.time()
    for step in tqdm(range(num_steps)):

        # Convert conservative to primitive
        #primitive_field = conservative_to_primitive(
        #    primitive_field,
        #    conservative_field,
        #    gamma=1.4,
        #)

        # Update the conservative field
        conservative_field = two_kernel_euler_update(
            primitive_field,
            primitive_dxyz_field,
            conservative_field,
            gamma=1.4,
            dt=dt,
        )

    # Compute MUPS
    wp.synchronize()
    toc = time.time()
    mups = nr_cells * num_steps / (toc - tic) / 1.0e6
    print(f"MUPS: {mups}")
