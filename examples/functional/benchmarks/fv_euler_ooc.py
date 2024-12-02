import time
import numpy as np
import cupy as cp
import warp as wp
from tqdm import tqdm
import argparse

wp.init()

from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.data import Fieldfloat32, Fielduint8
from pumpkin_pulse.operator import Operator
from pumpkin_pulse.operator.hydro.euler_eos import (
    ConservativeToPrimitive,
    TwoKernelEulerUpdate,
)

from out_of_core import OOCmap
from ooc_array import OOCArray

#comm = MPI.COMM_WORLD

# Argument parser for resolution
parser = argparse.ArgumentParser(description='Run a simple FV Euler simulation on the GPU.')
parser.add_argument('--resolution', type=int, default=256, help='Resolution of the simulation.')
parser.add_argument('--sub_resolution', type=int, default=128, help='Sub-resolution of the simulation.')
parser.add_argument('--nr_substeps', type=int, default=4, help='Number of substeps.')
parser.add_argument('--nr_compute_tiles', type=int, default=2, help='Number of compute tiles.')
resolution = parser.parse_args().resolution
sub_resolution = parser.parse_args().sub_resolution
nr_substeps = parser.parse_args().nr_substeps
nr_compute_tiles = parser.parse_args().nr_compute_tiles


class SplitCons(Operator):

    @wp.kernel
    def _split_cons(
        conservative_array: wp.array4d(dtype=wp.float32),
        conservative_field: Fieldfloat32,
    ):
        # Get index
        i, j, k = wp.tid()

        # Split the fields
        for c in range(5):
            conservative_field.data[c, i, j, k] = conservative_array[c, i, j, k]

    def __call__(
        self,
        conservative_array: wp.array4d(dtype=wp.float32),
        conservative_field: Fieldfloat32,
    ):
        # Launch kernel
        wp.launch(
            self._split_cons,
            inputs=[
                conservative_array,
                conservative_field,
            ],
            dim=conservative_field.shape,
        )

        return conservative_field

class CombineCons(Operator):

    @wp.kernel
    def _combine_cons(
        conservative_field: Fieldfloat32,
        conservative_array: wp.array4d(dtype=wp.float32),
    ):
        # Get index
        i, j, k = wp.tid()

        # Split the fields
        for c in range(5):
            conservative_array[c, i, j, k] = conservative_field.data[c, i, j, k]

    def __call__(
        self,
        conservative_field: Fieldfloat32,
        conservative_array: wp.array4d(dtype=wp.float32),
    ):
        # Launch kernel
        wp.launch(
            self._combine_cons,
            inputs=[
                conservative_field,
                conservative_array,
            ],
            dim=conservative_field.shape,
        )

        return conservative_array


if __name__ == '__main__':

    # Define simulation parameters
    dx = 1.0
    origin = (0.0, 0.0, 0.0)
    spacing = (dx, dx, dx)
    global_shape = (resolution, resolution, resolution)
    nr_cells = global_shape[0] * global_shape[1] * global_shape[2]
    sub_shape = (sub_resolution, sub_resolution, sub_resolution)

    # Electric parameters

    # Use CFL condition to determine time step
    dt = 0.0000001

    # Number of steps
    sub_steps = nr_substeps
    num_steps = 128
    global_steps = num_steps // sub_steps
    num_steps = global_steps * sub_steps
    print(f"Number of steps: {num_steps}")

    # Make the constructor
    constructor = Constructor()

    # Make the operators
    conservative_to_primitive = ConservativeToPrimitive()
    two_kernel_euler_update = TwoKernelEulerUpdate()
    split_cons = SplitCons()
    combine_cons = CombineCons()

    # Make the fields
    conservative_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=5,
        shape=[sub_resolution + 2*sub_steps]*3,
        origin=origin,
        spacing=spacing,
        ordering=0,
    )
    primitive_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=5,
        shape=[sub_resolution + 2*sub_steps]*3,
        origin=origin,
        spacing=spacing,
        ordering=0,
    )
    primitive_dxyz_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=15,
        shape=[sub_resolution + 2*sub_steps]*3,
        origin=origin,
        spacing=spacing,
        ordering=0,
    )

    # Make the fields on the host
    conservative_array = OOCArray(
        shape=[5] + list(global_shape),
        dtype=np.float32,
        tile_shape=[5] + list(sub_shape),
        padding=(0, sub_steps, sub_steps, sub_steps),
        #comm=comm,
        devices=[cp.cuda.Device(0)],
        codec=None,
        nr_compute_tiles=nr_compute_tiles,
    )

    # Run 1 step for kernel initialization
    conservative_field = two_kernel_euler_update(
        primitive_field,
        primitive_dxyz_field,
        conservative_field,
        gamma=1.4,
        dt=dt,
    )

    # FV Euler update
    @OOCmap((0,), backend="warp")
    def fv_euler_ooc_update(
            cons_array,
            conservative_field,
            primitive_field,
            primitive_dxyz_field,
    ):

        # Split the fields
        conservative_field = split_cons(cons_array, conservative_field)

        # Run the field updates
        for _ in range(sub_steps):
            # Update the conservative field
            conservative_field = two_kernel_euler_update(
                primitive_field,
                primitive_dxyz_field,
                conservative_field,
                gamma=1.4,
                dt=dt,
            )

        # Combine the fields
        cons_array = combine_cons(conservative_field, cons_array)

        return cons_array



    # Run the simulation
    wp.synchronize()
    tic = time.time()
    for step in tqdm(range(global_steps)):

        # Update the conservative field
        conservative_array = fv_euler_ooc_update(
            conservative_array,
            conservative_field,
            primitive_field,
            primitive_dxyz_field
        )

    # Compute MUPS
    wp.synchronize()
    toc = time.time()
    mups = nr_cells * num_steps / (toc - tic) / 1.0e6
    print(f"MUPS: {mups}")
