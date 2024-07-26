# Plane wave hitting a B2 Bomber

import os
import numpy as np
import warp as wp
from build123d import Rectangle, extrude, Sphere, Location, Circle, Rotation
from tqdm import tqdm

wp.init()

from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.data import Fieldfloat32, Fielduint8
from pumpkin_pulse.operator import Operator
from pumpkin_pulse.operator.electromagnetism import YeeElectricFieldUpdate, YeeMagneticFieldUpdate
from pumpkin_pulse.operator.mesh import StlToMesh, MeshToIdField
from pumpkin_pulse.operator.saver import FieldSaver

####### STL file ########
# B2 Spirit Stealth Bomber by JCanz on Thingiverse: https://www.thingiverse.com/thing:1048103
# This thing was created by Thingiverse user JCanz, and is licensed under Creative Commons - Attribution
#########################

class PlaneWaveInitialize(Operator):
    _c = wp.constant(3.0e8)
    _eps0 = wp.constant(8.854187817e-12)
    _mu0 = wp.constant(4.0 * 3.14159 * 1.0e-7)
    _eta0 = wp.constant(376.73031346177)

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
        x = electric_field.origin[0] + wp.float32(i) * electric_field.spacing[0]

        # Calculate wave number
        wave_number = 2.0 * 3.14159 * frequency / 3.0e8

        # Calculate the phase
        phase = 2.0 * 3.14159 * frequency * (x - center) / 3.0e8

        # Apply a gaussian envelope
        envelope = wp.exp(-0.5 * ((x - center) / sigma) ** 2.0)

        # Calculate the electric field
        electric_field.data[1, i, j, k] = amplitude * envelope * wp.cos(phase)

        # Calculate the magnetic field
        magnetic_field.data[2, i, j, k] = (amplitude / PlaneWaveInitialize._eta0) * envelope * wp.cos(phase)


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
 
if __name__ == '__main__':

    # Define simulation parameters
    dx = 0.1
    origin = (-25.0, -25.0, -25.0)
    spacing = (dx, dx, dx)
    shape = (int(50.0/dx), int(50.0/dx), int(50.0/dx))

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
    start_x = -20.0 # x location of the wave source
    frequency = 1.0e8
    amplitude = 1.0
    sigma = 1.0

    # Use CFL condition to determine time step
    simulation_time = 100.0 / c # time for wave to travel 100 meters
    dt = dx / (c * np.sqrt(3.0))
    num_steps = int(simulation_time / dt)
    print(f"Number of steps: {num_steps}")

    # Make output directory
    save_frequency = 4
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        spacing=spacing
    )
    electric_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    magnetic_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )

    # Make material property mappings
    eps_mapping = wp.from_numpy(np.array([eps, b2_eps], dtype=np.float32), dtype=wp.float32)
    mu_mapping = wp.from_numpy(np.array([mu, b2_mu], dtype=np.float32), dtype=wp.float32)
    sigma_e_mapping = wp.from_numpy(np.array([sigma_e, b2_sigma_e], dtype=np.float32), dtype=wp.float32)
    sigma_m_mapping = wp.from_numpy(np.array([sigma_m, b2_sigma_m], dtype=np.float32), dtype=wp.float32)

    # Make mesh
    mesh = stl_to_mesh("files/B2.stl")
    #mesh = stl_to_mesh("files/sphere.stl")
    #mesh = stl_to_mesh("files/combined_drivaer.stl")
    id_field = mesh_to_id_field(mesh, id_field, 1)

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

    # Run the simulation
    for step in tqdm(range(num_steps)):

        # Save the fields
        if step % save_frequency == 0:
            field_saver(electric_field, os.path.join(output_dir, f"electric_field_{str(step).zfill(4)}.vtk"))

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








