# Plane wave hitting a B2 Bomber

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
from pumpkin_pulse.operator.electromagnetism import YeeElectricFieldUpdate, YeeMagneticFieldUpdate

from out_of_core import OOCmap
from ooc_array import OOCArray



# Argument parser for resolution
parser = argparse.ArgumentParser(description='Run the Yee algorithm on a plane wave hitting a B2 Bomber.')
parser.add_argument('--resolution', type=int, default=1024, help='Resolution of the simulation.')
parser.add_argument('--sub_resolution', type=int, default=512, help='Sub-resolution of the simulation.')
parser.add_argument('--nr_substeps', type=int, default=8, help='Number of substeps.')
parser.add_argument('--nr_compute_tiles', type=int, default=1, help='Number of compute tiles.')
resolution = parser.parse_args().resolution
sub_resolution = parser.parse_args().sub_resolution
nr_substeps = parser.parse_args().nr_substeps
nr_compute_tiles = parser.parse_args().nr_compute_tiles


class SplitEH(Operator):

    @wp.kernel
    def _split_eh(
        eh: wp.array4d(dtype=wp.float32),
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
    ):
        # Get index
        i, j, k = wp.tid()

        # Split the fields
        for c in range(3):
            electric_field.data[c, i, j, k] = eh[c, i, j, k]
            magnetic_field.data[c, i, j, k] = eh[c + 3, i, j, k]

    def __call__(
        self,
        eh: wp.array4d(dtype=wp.float32),
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
    ):
        # Launch kernel
        wp.launch(
            self._split_eh,
            inputs=[
                eh,
                electric_field,
                magnetic_field,
            ],
            dim=electric_field.shape,
        )

        return electric_field, magnetic_field

class CombineEH(Operator):

    @wp.kernel
    def _combine_eh(
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
        eh: wp.array4d(dtype=wp.float32),
    ):
        # Get index
        i, j, k = wp.tid()

        # Split the fields
        for c in range(3):
            eh[c, i, j, k] = electric_field.data[c, i, j, k]
            eh[c + 3, i, j, k] = magnetic_field.data[c, i, j, k]

    def __call__(
        self,
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
        eh: wp.array4d(dtype=wp.float32),
    ):
        # Launch kernel
        wp.launch(
            self._combine_eh,
            inputs=[
                electric_field,
                magnetic_field,
                eh,
            ],
            dim=electric_field.shape,
        )

        return eh


if __name__ == '__main__':

    # Define simulation parameters
    dx = 1.0
    origin = (0.0, 0.0, 0.0)
    spacing = (dx, dx, dx)
    global_shape = (resolution, resolution, resolution)
    nr_cells = global_shape[0] * global_shape[1] * global_shape[2]
    sub_shape = (sub_resolution, sub_resolution, sub_resolution)

    # Electric parameters
    # Vacuum
    c = 3.0e8
    eps = 8.854187817e-12
    mu = 4.0 * wp.pi * 1.0e-7
    sigma_e = 0.0
    sigma_m = 0.0

    # Use CFL condition to determine time step
    dt = dx / (c * np.sqrt(3.0))

    # Number of steps
    sub_steps = nr_substeps
    num_steps = 512
    global_steps = num_steps // sub_steps
    num_steps = global_steps * sub_steps
    print(f"Number of steps: {num_steps}")

    # Make the constructor
    constructor = Constructor()

    # Make the operators
    e_field_update = YeeElectricFieldUpdate()
    h_field_update = YeeMagneticFieldUpdate()
    split_eh = SplitEH()
    combine_eh = CombineEH()

    # Make the fields on the device
    id_field = constructor.create_field(
        dtype=wp.uint8,
        cardinality=1,
        shape=sub_shape,
        origin=origin,
        spacing=spacing,
        ordering=0,
    )
    electric_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=sub_shape,
        origin=origin,
        spacing=spacing,
        ordering=0,
    )
    magnetic_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=sub_shape,
        origin=origin,
        spacing=spacing,
        ordering=0,
    )

    # Make material property mappings
    eps_mapping = wp.from_numpy(np.array([eps], dtype=np.float32), dtype=wp.float32)
    mu_mapping = wp.from_numpy(np.array([mu], dtype=np.float32), dtype=wp.float32)
    sigma_e_mapping = wp.from_numpy(np.array([sigma_e], dtype=np.float32), dtype=wp.float32)
    sigma_m_mapping = wp.from_numpy(np.array([sigma_m], dtype=np.float32), dtype=wp.float32)

    # Make the fields on the host
    eh_array = OOCArray(
        shape=[6] + list(global_shape),
        dtype=np.float32,
        tile_shape=[6] + list(sub_shape),
        padding=(0, sub_steps, sub_steps, sub_steps),
        devices=[cp.cuda.Device(0)],
        codec=None,
        nr_compute_tiles=nr_compute_tiles,
    )

    # Run 1 step for kernel initialization
    magnetic_field = h_field_update(
        electric_field,
        magnetic_field,
        id_field,
        mu_mapping,
        sigma_m_mapping,
        dt
    )
    electric_field = e_field_update(
        electric_field,
        magnetic_field,
        None,
        id_field,
        eps_mapping,
        sigma_e_mapping,
        dt
    )

    # EM OOC update
    @OOCmap((0,), backend="warp")
    def em_ooc_update(eh, electric_field, magnetic_field):

        # Split the fields
        electric_field, magnetic_field = split_eh(eh, electric_field, magnetic_field)

        # Run the field updates
        for _ in range(sub_steps):
            magnetic_field = h_field_update(
                electric_field,
                magnetic_field,
                id_field,
                mu_mapping,
                sigma_m_mapping,
                dt
            )
            electric_field = e_field_update(
                electric_field,
                magnetic_field,
                None,
                id_field,
                eps_mapping,
                sigma_e_mapping,
                dt
            )

        # Combine the fields
        eh = combine_eh(electric_field, magnetic_field, eh)

        return eh

    # Run the simulation
    wp.synchronize()
    tic = time.time()
    for step in tqdm(range(global_steps)):

        # Run the OOC update
        eh_array = em_ooc_update(eh_array, electric_field, magnetic_field)

    # Compute MUPS
    wp.synchronize()
    toc = time.time()
    mups = nr_cells * num_steps / (toc - tic) / 1.0e6
    print(f"MUPS: {mups}")
