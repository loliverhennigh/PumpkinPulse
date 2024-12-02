# Plane wave hitting a B2 Bomber

import os
import numpy as np
import warp as wp
from build123d import Rectangle, extrude, Sphere, Location, Circle, Rotation
from tqdm import tqdm
import matplotlib.pyplot as plt

wp.init()

from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.data import Fieldfloat32, Fielduint8
from pumpkin_pulse.operator import Operator
from pumpkin_pulse.operator.electromagnetism import YeeElectricFieldUpdate, YeeMagneticFieldUpdate
from pumpkin_pulse.operator.mesh import StlToMesh, MeshToIdField
from pumpkin_pulse.operator.saver import FieldSaver

from convergence import convergence_analysis

####### STL file ########
# B2 Spirit Stealth Bomber by JCanz on Thingiverse: https://www.thingiverse.com/thing:1048103
# This thing was created by Thingiverse user JCanz, and is licensed under Creative Commons - Attribution
#########################

class PlaneWaveInitialize(Operator):

    @wp.kernel
    def _initialize_plane_wave(
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
        frequency: float,
        center: float,
        amplitude: float,
        sigma: float,
    ):
        # Get index
        i, j, k = wp.tid()

        # Get the x location of the current cell
        elec_x = electric_field.origin[0] + wp.float32(i) * electric_field.spacing[0] - electric_field.spacing[0] / 2.0
        mag_x = electric_field.origin[0] + wp.float32(i) * electric_field.spacing[0] + electric_field.spacing[0] / 2.0

        # Calculate wave number
        wave_number = 2.0 * 3.14159 * frequency / 3.0e8

        # Calculate the phase
        elec_phase = 2.0 * 3.14159 * frequency * (elec_x - center) / 3.0e8
        mag_phase = 2.0 * 3.14159 * frequency * (mag_x - center) / 3.0e8

        # Apply a gaussian envelope
        elec_envelope = wp.exp(-0.5 * ((elec_x - center) / sigma) ** 2.0)
        mag_envelope = wp.exp(-0.5 * ((mag_x - center) / sigma) ** 2.0)

        # Calculate the electric field
        electric_field.data[1, i, j, k] = amplitude * elec_envelope * wp.cos(elec_phase)

        # Calculate the magnetic field
        magnetic_field.data[2, i, j, k] = (amplitude / 1.0) * mag_envelope * wp.cos(mag_phase)


    def __call__(
        self,
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
        frequency: float,
        center: float,
        amplitude: float,
        sigma: float,
    ):
        # Launch kernel
        wp.launch(
            self._initialize_plane_wave,
            inputs=[
                electric_field,
                magnetic_field,
                frequency,
                center,
                amplitude,
                sigma,
            ],
            dim=electric_field.shape,
        )

        return electric_field

def run_sim(factor):

    # Define simulation parameters
    dx = 1.0 / factor
    origin = (-25.0, -25.0, -25.0)
    spacing = (dx, dx, dx)
    shape = (int(50.0/dx), int(50.0/dx), int(50.0/dx))
    nr_cells = shape[0] * shape[1] * shape[2]

    # Electric parameters
    # Vacuum
    c = 3.0e8
    eps = 8.854187817e-12
    mu = 4.0 * wp.pi * 1.0e-7
    sigma_e = 0.0
    sigma_m = 0.0

    # B2 Bomber (copper)
    b2_eps = eps
    b2_mu = mu
    b2_sigma_e=5.96e7
    b2_sigma_m = 0.0

    # Wave parameters (2 GHz)
    start_x = 0.0 # x location of the wave source
    frequency = 3.0e6
    amplitude = 1.0
    sigma = 5.0

    # Use CFL condition to determine time step
    dt = dx / (c * np.sqrt(3.0)) / 10.0

    # Make the constructor
    constructor = Constructor()

    # Make the operators
    e_field_update = YeeElectricFieldUpdate()
    h_field_update = YeeMagneticFieldUpdate()
    plane_wave_initialize = PlaneWaveInitialize()
    field_saver = FieldSaver()
    stl_to_mesh = StlToMesh()
    mesh_to_id_field = MeshToIdField()

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
    eps_mapping = wp.from_numpy(np.array([eps, b2_eps], dtype=np.float32), dtype=wp.float32)
    mu_mapping = wp.from_numpy(np.array([mu, b2_mu], dtype=np.float32), dtype=wp.float32)
    sigma_e_mapping = wp.from_numpy(np.array([sigma_e, b2_sigma_e], dtype=np.float32), dtype=wp.float32)
    sigma_m_mapping = wp.from_numpy(np.array([sigma_m, b2_sigma_m], dtype=np.float32), dtype=wp.float32)

    # Make mesh
    mesh = stl_to_mesh("files/B2.stl")
    #id_field = mesh_to_id_field(mesh, id_field, 1)

    # Save id field
    field_saver(id_field, os.path.join(output_dir, "id_field.vtk"))

    # Initialize the electric and magnetic fields
    electric_field = plane_wave_initialize(
        electric_field,
        magnetic_field,
        frequency,
        start_x,
        amplitude,
        sigma,
    )

    # Run half time step to align with Yee grid on multpile resolutions
    electric_field = e_field_update(
        electric_field,
        magnetic_field,
        None,
        id_field,
        eps_mapping,
        sigma_e_mapping,
        dt / 2.0
    )

    # Run the simulation
    import time
    tic = time.time()
    for step in tqdm(range(factor)):

        # Update the magnetic field
        magnetic_field = h_field_update(
            electric_field,
            magnetic_field,
            id_field,
            mu_mapping,
            sigma_m_mapping,
            dt
        )

        # Update the electric field (skip last update)
        if (step+1) % factor == 0:
            print("Last step")
            print(step)
            break
        else:
            electric_field = e_field_update(
                electric_field,
                magnetic_field,
                None,
                id_field,
                eps_mapping,
                sigma_e_mapping,
                dt
            )

    # Take the second half step
    electric_field = e_field_update(
        electric_field,
        magnetic_field,
        None,
        id_field,
        eps_mapping,
        sigma_e_mapping,
        dt / 2.0
    )

    # Sync the fields
    wp.synchronize()

    # Plot final field
    plt.figure()
    plt.imshow(electric_field.data[1, :, 0, :].numpy())
    plt.colorbar()
    plt.title("Electric field")
    plt.savefig(os.path.join(output_dir, f"electric_field_{factor}.png"))

    plt.figure()
    plt.imshow(magnetic_field.data[2, :, 0, :].numpy())
    plt.colorbar()
    plt.title("Magnetic field")
    plt.savefig(os.path.join(output_dir, f"magnetic_field_{factor}.png"))

    yield {
        "electric_field": electric_field.data.numpy(),
        "magnetic_field": magnetic_field.data.numpy(),
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
