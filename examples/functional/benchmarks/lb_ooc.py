# Description: This file contains a simple example of using the OOCmap
# decorator to apply a function to a distributed array.
# Solves Lattice Boltzmann Taylor Green vortex decay

import time
import warp as wp
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cupy as cp
import time
from tqdm import tqdm
import argparse

from out_of_core import OOCmap
from ooc_array import OOCArray

# Disable pinned memory allocator
cp.cuda.set_pinned_memory_allocator(None)

# Initialize Warp
wp.init()

@wp.func
def sample_f(
    f: wp.array4d(dtype=float),
    q: int,
    x: int,
    y: int,
    z: int,
    width: int,
    height: int,
    length: int,
):

    # Periodic boundary conditions
    if x == -1:
        x = width - 1
    if x == width:
        x = 0
    if y == -1:
        y = height - 1
    if y == height:
        y = 0
    if z == -1:
        z = length - 1
    if z == length:
        z = 0
    s = f[q, x, y, z]
    return s

@wp.kernel
def stream_collide(
    f0: wp.array4d(dtype=float),
    f1: wp.array4d(dtype=float),
    width: int,
    height: int,
    length: int,
    tau: float,
):

    # get index
    x, y, z = wp.tid()

    # sample needed points
    f_1_1_1 = sample_f(f0,  0,     x,     y,     z, width, height, length)
    f_2_1_1 = sample_f(f0,  1, x - 1,     y,     z, width, height, length)
    f_0_1_1 = sample_f(f0,  2, x + 1,     y,     z, width, height, length)
    f_1_2_1 = sample_f(f0,  3,     x, y - 1,     z, width, height, length)
    f_1_0_1 = sample_f(f0,  4,     x, y + 1,     z, width, height, length)
    f_1_1_2 = sample_f(f0,  5,     x,     y, z - 1, width, height, length)
    f_1_1_0 = sample_f(f0,  6,     x,     y, z + 1, width, height, length)
    f_1_2_2 = sample_f(f0,  7,     x, y - 1, z - 1, width, height, length)
    f_1_0_0 = sample_f(f0,  8,     x, y + 1, z + 1, width, height, length)
    f_1_2_0 = sample_f(f0,  9,     x, y - 1, z + 1, width, height, length)
    f_1_0_2 = sample_f(f0, 10,     x, y + 1, z - 1, width, height, length)
    f_2_1_2 = sample_f(f0, 11, x - 1,     y, z - 1, width, height, length)
    f_0_1_0 = sample_f(f0, 12, x + 1,     y, z + 1, width, height, length)
    f_2_1_0 = sample_f(f0, 13, x - 1,     y, z + 1, width, height, length)
    f_0_1_2 = sample_f(f0, 14, x + 1,     y, z - 1, width, height, length)
    f_2_2_1 = sample_f(f0, 15, x - 1, y - 1,     z, width, height, length)
    f_0_0_1 = sample_f(f0, 16, x + 1, y + 1,     z, width, height, length)
    f_2_0_1 = sample_f(f0, 17, x - 1, y + 1,     z, width, height, length)
    f_0_2_1 = sample_f(f0, 18, x + 1, y - 1,     z, width, height, length)

    # compute u and p
    p = (f_1_1_1
       + f_2_1_1 + f_0_1_1
       + f_1_2_1 + f_1_0_1
       + f_1_1_2 + f_1_1_0
       + f_1_2_2 + f_1_0_0
       + f_1_2_0 + f_1_0_2
       + f_2_1_2 + f_0_1_0
       + f_2_1_0 + f_0_1_2
       + f_2_2_1 + f_0_0_1
       + f_2_0_1 + f_0_2_1)
    u = (f_2_1_1 - f_0_1_1
       + f_2_1_2 - f_0_1_0
       + f_2_1_0 - f_0_1_2
       + f_2_2_1 - f_0_0_1
       + f_2_0_1 - f_0_2_1)
    v = (f_1_2_1 - f_1_0_1
       + f_1_2_2 - f_1_0_0
       + f_1_2_0 - f_1_0_2
       + f_2_2_1 - f_0_0_1
       - f_2_0_1 + f_0_2_1)
    w = (f_1_1_2 - f_1_1_0
       + f_1_2_2 - f_1_0_0
       - f_1_2_0 + f_1_0_2
       + f_2_1_2 - f_0_1_0
       - f_2_1_0 + f_0_1_2)
    res_p = 1.0 / p
    u = u * res_p
    v = v * res_p
    w = w * res_p
    uxu = u * u + v * v + w * w

    # compute e dot u
    exu_1_1_1 = 0
    exu_2_1_1 = u
    exu_0_1_1 = -u
    exu_1_2_1 = v
    exu_1_0_1 = -v
    exu_1_1_2 = w
    exu_1_1_0 = -w
    exu_1_2_2 = v + w
    exu_1_0_0 = -v - w
    exu_1_2_0 = v - w
    exu_1_0_2 = -v + w
    exu_2_1_2 = u + w
    exu_0_1_0 = -u - w
    exu_2_1_0 = u - w
    exu_0_1_2 = -u + w
    exu_2_2_1 = u + v
    exu_0_0_1 = -u - v
    exu_2_0_1 = u - v
    exu_0_2_1 = -u + v


    # compute equilibrium dist
    factor_1 = 1.5
    factor_2 = 4.5
    weight_0 = 0.33333333
    weight_1 = 0.05555555
    weight_2 = 0.02777777
    f_eq_1_1_1 = weight_0 * (p * (factor_1 * (- uxu) + 1.0))
    f_eq_2_1_1 = weight_1 * (p * (factor_1 * (2.0 * exu_2_1_1 - uxu) + factor_2 * (exu_2_1_1 * exu_2_1_1) + 1.0))
    f_eq_0_1_1 = weight_1 * (p * (factor_1 * (2.0 * exu_0_1_1 - uxu) + factor_2 * (exu_0_1_1 * exu_0_1_1) + 1.0))
    f_eq_1_2_1 = weight_1 * (p * (factor_1 * (2.0 * exu_1_2_1 - uxu) + factor_2 * (exu_1_2_1 * exu_1_2_1) + 1.0))
    f_eq_1_0_1 = weight_1 * (p * (factor_1 * (2.0 * exu_1_0_1 - uxu) + factor_2 * (exu_1_2_1 * exu_1_2_1) + 1.0))
    f_eq_1_1_2 = weight_1 * (p * (factor_1 * (2.0 * exu_1_1_2 - uxu) + factor_2 * (exu_1_1_2 * exu_1_1_2) + 1.0))
    f_eq_1_1_0 = weight_1 * (p * (factor_1 * (2.0 * exu_1_1_0 - uxu) + factor_2 * (exu_1_1_0 * exu_1_1_0) + 1.0))
    f_eq_1_2_2 = weight_2 * (p * (factor_1 * (2.0 * exu_1_2_2 - uxu) + factor_2 * (exu_1_2_2 * exu_1_2_2) + 1.0))
    f_eq_1_0_0 = weight_2 * (p * (factor_1 * (2.0 * exu_1_0_0 - uxu) + factor_2 * (exu_1_0_0 * exu_1_0_0) + 1.0))
    f_eq_1_2_0 = weight_2 * (p * (factor_1 * (2.0 * exu_1_2_0 - uxu) + factor_2 * (exu_1_2_0 * exu_1_2_0) + 1.0))
    f_eq_1_0_2 = weight_2 * (p * (factor_1 * (2.0 * exu_1_0_2 - uxu) + factor_2 * (exu_1_0_2 * exu_1_0_2) + 1.0))
    f_eq_2_1_2 = weight_2 * (p * (factor_1 * (2.0 * exu_2_1_2 - uxu) + factor_2 * (exu_2_1_2 * exu_2_1_2) + 1.0))
    f_eq_0_1_0 = weight_2 * (p * (factor_1 * (2.0 * exu_0_1_0 - uxu) + factor_2 * (exu_0_1_0 * exu_0_1_0) + 1.0))
    f_eq_2_1_0 = weight_2 * (p * (factor_1 * (2.0 * exu_2_1_0 - uxu) + factor_2 * (exu_2_1_0 * exu_2_1_0) + 1.0))
    f_eq_0_1_2 = weight_2 * (p * (factor_1 * (2.0 * exu_0_1_2 - uxu) + factor_2 * (exu_0_1_2 * exu_0_1_2) + 1.0))
    f_eq_2_2_1 = weight_2 * (p * (factor_1 * (2.0 * exu_2_2_1 - uxu) + factor_2 * (exu_2_2_1 * exu_2_2_1) + 1.0))
    f_eq_0_0_1 = weight_2 * (p * (factor_1 * (2.0 * exu_0_0_1 - uxu) + factor_2 * (exu_0_0_1 * exu_0_0_1) + 1.0))
    f_eq_2_0_1 = weight_2 * (p * (factor_1 * (2.0 * exu_2_0_1 - uxu) + factor_2 * (exu_2_0_1 * exu_2_0_1) + 1.0))
    f_eq_0_2_1 = weight_2 * (p * (factor_1 * (2.0 * exu_0_2_1 - uxu) + factor_2 * (exu_0_2_1 * exu_0_2_1) + 1.0))

    # set next lattice state
    inv_tau = (1.0 / tau)
    f1[0, x, y, z] =  f_1_1_1 - inv_tau * (f_1_1_1 - f_eq_1_1_1)
    f1[1, x, y, z] =  f_2_1_1 - inv_tau * (f_2_1_1 - f_eq_2_1_1)
    f1[2, x, y, z] =  f_0_1_1 - inv_tau * (f_0_1_1 - f_eq_0_1_1)
    f1[3, x, y, z] =  f_1_2_1 - inv_tau * (f_1_2_1 - f_eq_1_2_1)
    f1[4, x, y, z] =  f_1_0_1 - inv_tau * (f_1_0_1 - f_eq_1_0_1)
    f1[5, x, y, z] =  f_1_1_2 - inv_tau * (f_1_1_2 - f_eq_1_1_2)
    f1[6, x, y, z] =  f_1_1_0 - inv_tau * (f_1_1_0 - f_eq_1_1_0)
    f1[7, x, y, z] =  f_1_2_2 - inv_tau * (f_1_2_2 - f_eq_1_2_2)
    f1[8, x, y, z] =  f_1_0_0 - inv_tau * (f_1_0_0 - f_eq_1_0_0)
    f1[9, x, y, z] =  f_1_2_0 - inv_tau * (f_1_2_0 - f_eq_1_2_0)
    f1[10, x, y, z] = f_1_0_2 - inv_tau * (f_1_0_2 - f_eq_1_0_2)
    f1[11, x, y, z] = f_2_1_2 - inv_tau * (f_2_1_2 - f_eq_2_1_2)
    f1[12, x, y, z] = f_0_1_0 - inv_tau * (f_0_1_0 - f_eq_0_1_0)
    f1[13, x, y, z] = f_2_1_0 - inv_tau * (f_2_1_0 - f_eq_2_1_0)
    f1[14, x, y, z] = f_0_1_2 - inv_tau * (f_0_1_2 - f_eq_0_1_2)
    f1[15, x, y, z] = f_2_2_1 - inv_tau * (f_2_2_1 - f_eq_2_2_1)
    f1[16, x, y, z] = f_0_0_1 - inv_tau * (f_0_0_1 - f_eq_0_0_1)
    f1[17, x, y, z] = f_2_0_1 - inv_tau * (f_2_0_1 - f_eq_2_0_1)
    f1[18, x, y, z] = f_0_2_1 - inv_tau * (f_0_2_1 - f_eq_0_2_1)



@OOCmap((0,), backend="warp")
def apply_stream_collide(f0, f1, tau: float, nr_steps: int):

    # Apply streaming and collision for nr_steps
    shape = f0.shape
    for _ in range(nr_steps):
        # Apply streaming and collision step
        wp.launch(
            kernel=stream_collide,
            dim=shape[1:],
            inputs=[f0, f1, shape[1], shape[2], shape[3], tau],
            device=f1.device,
        )

        # Swap f0 and f1
        f0, f1 = f1, f0

    return f0

# Argument parser for resolution
parser = argparse.ArgumentParser(description='Run the Yee algorithm on a plane wave hitting a B2 Bomber.')
parser.add_argument('--resolution', type=int, default=256, help='Resolution of the simulation.')
parser.add_argument('--sub_resolution', type=int, default=128, help='Sub-resolution of the simulation.')
parser.add_argument('--nr_substeps', type=int, default=16, help='Number of substeps.')
parser.add_argument('--nr_compute_tiles', type=int, default=2, help='Number of compute tiles.')
resolution = parser.parse_args().resolution
sub_resolution = parser.parse_args().sub_resolution
nr_substeps = parser.parse_args().nr_substeps
nr_compute_tiles = parser.parse_args().nr_compute_tiles

if __name__ == "__main__":

    # Sim Parameters
    n = resolution
    sub_n = sub_resolution
    tau = 0.505
    nr_sub_steps = nr_substeps
    dx = 2.0 * np.pi / n

    # Make OOC distributed array
    f0 = OOCArray(
        shape=(19, n, n, n),
        dtype=np.float32,
        tile_shape=(19, sub_n, sub_n, sub_n),
        padding=(0, nr_sub_steps, nr_sub_steps, nr_sub_steps),
        devices=[cp.cuda.Device(0)],
        codec=None,
        nr_compute_tiles=nr_compute_tiles,
    )

    # Make f1
    f1 = wp.empty(
        (
            19,
            sub_n + 2 * nr_sub_steps,
            sub_n + 2 * nr_sub_steps,
            sub_n + 2 * nr_sub_steps,
        ),
        dtype=wp.float32,
        device="cuda:0",
    )

    # Apply streaming and collision
    nr_steps = 16
    step = []
    comp_ratio = []
    t0 = time.time()
    for _ in tqdm(range(nr_steps)):
        f0 = apply_stream_collide(f0, f1, tau, nr_sub_steps)

        #if _ % 4 == 0:
        #    # Create a new figure
        #    plt.figure(figsize=(15, 8))

        #    # Plot compression ratio
        #    plt.subplot(1, 2, 1)
        #    plt.title("Compression Ratio Taylor Green, codec: " + codec_name + ", tau: " + str(tau) + ", n^3: " + str(n))
        #    plt.plot(step, comp_ratio)
        #    plt.xlabel("Step")
        #    plt.ylabel("Compression Ratio")

        #    # Plot results
        #    plt.subplot(1, 2, 2)
        #    np_f = f0.get_array()
        #    plt.title("f_3, step: " + str(_ * nr_sub_steps))
        #    plt.imshow(np_f[3, :, :, 200], cmap="jet")
        #    plt.colorbar()
        #    plt.tight_layout()

        #    # Save figure
        #    plt.savefig("f_snappy_" + str(_ * nr_sub_steps).zfill(5) + ".png")
        #    plt.close()

    cp.cuda.Stream.null.synchronize()
    t1 = time.time()

    # Compute MLUPS
    mlups = (nr_sub_steps * nr_steps * n * n * n) / (t1 - t0) / 1e6
    print("Nr Million Cells: ", n * n * n / 1e6)
    print("MLUPS: ", mlups)
