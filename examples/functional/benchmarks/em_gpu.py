# Plane wave hitting a B2 Bomber

import time
import numpy as np
import warp as wp
from tqdm import tqdm
import argparse

wp.init()

from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.data import Fieldfloat32, Fielduint8
from pumpkin_pulse.operator.electromagnetism import YeeElectricFieldUpdate, YeeMagneticFieldUpdate

# Argument parser for resolution
parser = argparse.ArgumentParser(description='Run the Yee algorithm on a plane wave hitting a B2 Bomber.')
parser.add_argument('--resolution', type=int, default=512, help='Resolution of the simulation.')
resolution = parser.parse_args().resolution

if __name__ == '__main__':

    # Define simulation parameters
    dx = 1.0
    origin = (0.0, 0.0, 0.0)
    spacing = (dx, dx, dx)
    shape = (resolution, resolution, resolution)
    nr_cells = shape[0] * shape[1] * shape[2]

    # Electric parameters
    # Vacuum
    c = 3.0e8
    eps = 8.854187817e-12
    mu = 4.0 * wp.pi * 1.0e-7
    sigma_e = 0.0
    sigma_m = 0.0

    # Use CFL condition to determine time step
    dt = dx / (c * np.sqrt(3.0))
    num_steps = 256
    print(f"Number of steps: {num_steps}")

    # Make the constructor
    constructor = Constructor()

    # Make the operators
    e_field_update = YeeElectricFieldUpdate()
    h_field_update = YeeMagneticFieldUpdate()

    # Make the fields
    id_field = constructor.create_field(
        dtype=wp.uint8,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing,
        ordering=0,
    )
    electric_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing,
        ordering=0,
    )
    magnetic_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing,
        ordering=0,
    )

    # Make material property mappings
    eps_mapping = wp.from_numpy(np.array([eps], dtype=np.float32), dtype=wp.float32)
    mu_mapping = wp.from_numpy(np.array([mu], dtype=np.float32), dtype=wp.float32)
    sigma_e_mapping = wp.from_numpy(np.array([sigma_e], dtype=np.float32), dtype=wp.float32)
    sigma_m_mapping = wp.from_numpy(np.array([sigma_m], dtype=np.float32), dtype=wp.float32)

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

    # Run the simulation
    wp.synchronize()
    tic = time.time()
    for step in tqdm(range(num_steps)):

        # Update the magnetic field
        magnetic_field = h_field_update(
            electric_field,
            magnetic_field,
            id_field,
            mu_mapping,
            sigma_m_mapping,
            dt
        )

        # Update the electric
        electric_field = e_field_update(
            electric_field,
            magnetic_field,
            None,
            id_field,
            eps_mapping,
            sigma_e_mapping,
            dt
        )

    # Compute MUPS
    wp.synchronize()
    toc = time.time()
    mups = nr_cells * num_steps / (toc - tic) / 1.0e6
    print(f"MUPS: {mups}")
