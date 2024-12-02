# Plane wave hitting a B2 Bomber

import os
import numpy as np
import warp as wp
import matplotlib.pyplot as plt
from build123d import Rectangle, extrude, Sphere, Location, Circle, Rotation
from tqdm import tqdm

wp.init()

from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.data import Fieldfloat32, Fielduint8
from pumpkin_pulse.operator import Operator
from pumpkin_pulse.operator.electromagnetism import (
    YeeElectricFieldUpdate,
    YeeMagneticFieldUpdate,
    InitializePML,
    PMLElectricFieldUpdate,
    PMLMagneticFieldUpdate,
    PMLPhiEUpdate,
    PMLPhiHUpdate,
)
from pumpkin_pulse.operator.mesh import StlToMesh, MeshToIdField
from pumpkin_pulse.operator.saver import FieldSaver

####### STL file ########
# B2 Spirit Stealth Bomber by JCanz on Thingiverse: https://www.thingiverse.com/thing:1048103
# This thing was created by Thingiverse user JCanz, and is licensed under Creative Commons - Attribution
#########################

class PlaneWaveInitialize(Operator):

    @wp.kernel
    def _initialize_plane_wave(
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
        wavelength: float,
        center: float,
        amplitude: float,
        sigma: float,
    ):
        # Get index
        i, j, k = wp.tid()

        # Get the x location of the current cell
        x = electric_field.origin[0] + wp.float32(i) * electric_field.spacing[0]
        y = electric_field.origin[1] + wp.float32(j) * electric_field.spacing[1]
        z = electric_field.origin[2] + wp.float32(k) * electric_field.spacing[2]

        # Calculate wave number
        wave_number = 1.0 / wavelength

        # Calculate the phase
        phase_0 = (1.0 / wavelength) * (x - center)
        phase_1 = 2.0 * (1.0 / wavelength) * (x - center)
        phase_2 = 3.0 * (1.0 / wavelength) * (x - center)
        phase_3 = 4.0 * (1.0 / wavelength) * (x - center)
        phase_4 = 5.0 * (1.0 / wavelength) * (x - center)
        phase_5 = 6.0 * (1.0 / wavelength) * (x - center)

        # Apply a gaussian envelope
        envelope = wp.exp(-0.5 * (wp.sqrt((x - center)**2.0)  / sigma) ** 2.0)

        # coef
        coef_0 = 1.0
        coef_1 = 0.5
        coef_2 = -0.8
        coef_3 = 0.3
        coef_4 = -0.6
        coef_5 = -0.2

        # Calculate the electric field
        electric_field.data[1, i, j, k] = (
            amplitude
            * envelope
            * (
                wp.cos(phase_0) * coef_0
                + wp.cos(phase_1) * coef_1
                + wp.cos(phase_2) * coef_2
                + wp.cos(phase_3) * coef_3
                + wp.cos(phase_4) * coef_4
                + wp.cos(phase_5) * coef_5
            )
        )

        # Calculate the magnetic field
        magnetic_field.data[2, i, j, k] = (
            amplitude
            * envelope
            * (
                wp.cos(phase_0) * coef_0
                + wp.cos(phase_1) * coef_1
                + wp.cos(phase_2) * coef_2
                + wp.cos(phase_3) * coef_3
                + wp.cos(phase_4) * coef_4
                + wp.cos(phase_5) * coef_5
            )
        )


    def __call__(
        self,
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
        wavelength: float,
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
                wavelength,
                center,
                amplitude,
                sigma,
            ],
            dim=electric_field.shape,
        )

        return electric_field
 
if __name__ == '__main__':

    # Define simulation parameters
    dx = 0.02
    origin = (-25.0, -25.0, 0.0)
    spacing = (dx, dx, dx)
    shape = (int(50.0/dx), int(50.0/dx), 1)
    nr_cells = shape[0] * shape[1] * shape[2]
    print(f"Number of cells: {nr_cells}")

    # Electric parameters
    # Vacuum
    eps = 1.0 #8.854187817e-12
    mu = 1.0 # 4.0 * wp.pi * 1.0e-7
    c = 1.0 / np.sqrt(eps * mu)
    sigma_e = 0.0
    sigma_m = 0.0

    # B2 Bomber (copper)
    b2_eps = eps
    b2_mu = mu
    b2_sigma_e = 5.96e7
    b2_sigma_m = 0.0

    # Wave parameters (2 GHz)
    start_x = 0.0 # x location of the wave source
    #frequency = 2.5e7
    amplitude = 1.0
    wavelength = 1.0
    sigma = 3.0 * wavelength

    # PML parameters
    pml_width = 10

    # Use CFL condition to determine time step
    simulation_time = 35.0 / c # time for wave to travel 100 meters
    courant_number = 1.0 / np.sqrt(2.0) * 0.99
    dt = courant_number * (dx / c)
    num_steps = int(simulation_time / dt)
    print(f"Number of steps: {num_steps}")

    # Make output directory
    save_frequency = 4
    output_dir = f"output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Make the operators
    e_field_update = YeeElectricFieldUpdate()
    h_field_update = YeeMagneticFieldUpdate()
    initialize_pml_layer = InitializePML()
    pml_e_field_update = PMLElectricFieldUpdate()
    pml_h_field_update = PMLMagneticFieldUpdate()
    pml_phi_e_update = PMLPhiEUpdate()
    pml_phi_h_update = PMLPhiHUpdate()
    plane_wave_initialize = PlaneWaveInitialize()
    field_saver = FieldSaver()
    stl_to_mesh = StlToMesh()
    mesh_to_id_field = MeshToIdField()

    # Make the constructor
    constructor = Constructor(
        shape=shape,
        origin=origin,
        spacing=spacing,
    )

    # Make the fields
    id_field = constructor.create_field(
        dtype=wp.uint8,
        cardinality=1,
    )
    electric_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
    )
    magnetic_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
    )
    pml_layer_lower_x = constructor.create_field(
        dtype=wp.float32,
        cardinality=36,
        offset=(0, 0, 0),
        shape=(pml_width, shape[1], shape[2]),
    )
    pml_layer_upper_x = constructor.create_field(
        dtype=wp.float32,
        cardinality=36,
        offset=(shape[0] - pml_width, 0, 0),
        shape=(pml_width, shape[1], shape[2]),
    )
    #pml_layer_lower_y = constructor.create_field(
    #    dtype=wp.float32,
    #    cardinality=36,
    #    offset=(0, 0, 0),
    #    shape=(shape[0], pml_width, shape[2]),
    #)
    #pml_layer_upper_y = constructor.create_field(
    #    dtype=wp.float32,
    #    cardinality=36,
    #    offset=(0, shape[1] - pml_width, 0),
    #    shape=(shape[0], pml_width, shape[2]),
    #)
    #pml_layer_lower_z = constructor.create_field(
    #    dtype=wp.float32,
    #    cardinality=36,
    #    offset=(0, 0, 0),
    #    shape=(shape[0], shape[1], pml_width),
    #)
    #pml_layer_upper_z = constructor.create_field(
    #    dtype=wp.float32,
    #    cardinality=36,
    #    offset=(0, 0, shape[2] - pml_width),
    #    shape=(shape[0], shape[1], pml_width),
    #)

    # Make material property mappings
    eps_mapping = wp.from_numpy(np.array([eps, b2_eps], dtype=np.float32), dtype=wp.float32)
    mu_mapping = wp.from_numpy(np.array([mu, b2_mu], dtype=np.float32), dtype=wp.float32)
    sigma_e_mapping = wp.from_numpy(np.array([sigma_e, b2_sigma_e], dtype=np.float32), dtype=wp.float32)
    sigma_m_mapping = wp.from_numpy(np.array([sigma_m, b2_sigma_m], dtype=np.float32), dtype=wp.float32)

    ## Make mesh
    #mesh = stl_to_mesh("files/B2.stl")
    ##mesh = stl_to_mesh("files/sphere.stl")
    ##mesh = stl_to_mesh("files/combined_drivaer.stl")
    #id_field = mesh_to_id_field(mesh, id_field, 1)

    ## Save id field
    #field_saver(id_field, os.path.join(output_dir, "id_field.vtk"))

    # Initialize the electric and magnetic fields
    electric_field = plane_wave_initialize(
        electric_field,
        magnetic_field,
        wavelength,
        start_x,
        amplitude,
        sigma,
    )

    # Initialize the PML layers
    pml_layer_lower_x = initialize_pml_layer(
        pml_layer_lower_x,
        direction=wp.vec3f(1.0, 0.0, 0.0),
        thickness=pml_width,
        courant_number=courant_number,
    )
    pml_layer_upper_x = initialize_pml_layer(
        pml_layer_upper_x,
        direction=wp.vec3f(-1.0, 0.0, 0.0),
        thickness=pml_width,
        courant_number=courant_number,
    )
    #pml_layer_lower_y = initialize_pml_layer(
    #    pml_layer_lower_y,
    #    direction=wp.vec3f(0.0, 1.0, 0.0),
    #    thickness=pml_width,
    #    courant_number=courant_number,
    #)
    #pml_layer_upper_y = initialize_pml_layer(
    #    pml_layer_upper_y,
    #    direction=wp.vec3f(0.0, -1.0, 0.0),
    #    thickness=pml_width,
    #    courant_number=courant_number,
    #)
    #pml_layer_lower_z = initialize_pml_layer(
    #    pml_layer_lower_z,
    #    direction=wp.vec3f(0.0, 0.0, 1.0),
    #    thickness=pml_width,
    #    courant_number=courant_number,
    #)
    #pml_layer_upper_z = initialize_pml_layer(
    #    pml_layer_upper_z,
    #    direction=wp.vec3f(0.0, 0.0, -1.0),
    #    thickness=pml_width,
    #    courant_number=courant_number
    #)
    pml_layers = [
        pml_layer_lower_x,
        pml_layer_upper_x,
        #pml_layer_lower_y,
        #pml_layer_upper_y,
        #pml_layer_lower_z,
        #pml_layer_upper_z,
    ]

    # Run the simulation
    import time
    tic = time.time()
    for step in tqdm(range(num_steps)):

        # Save the fields
        if step % save_frequency == 0:
            field_saver(
                {"electric_field": electric_field, "magnetic_field": magnetic_field},
                os.path.join(output_dir, f"fields_{str(step).zfill(4)}.vtk"),
            )
            for _, pml_layer in enumerate(pml_layers):
                side = "abcdef"[_]
                field_saver(
                    {"pml_layer": pml_layer},
                    os.path.join(output_dir, f"pml_{side}_{str(step).zfill(4)}.vtk"),
                )

        # Update the PML phi_e fields
        for pml_layer in pml_layers:
            pml_layer = pml_phi_e_update(
                magnetic_field,
                pml_layer,
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

        # Update the electric field with PML
        for pml_layer in pml_layers:
            electric_field = pml_e_field_update(
                electric_field,
                pml_layer,
                id_field,
                eps_mapping,
                dt,
            )

        # Update the PML phi_h fields
        for pml_layer in pml_layers:
            pml_layer = pml_phi_h_update(
                electric_field,
                pml_layer,
            )

        # Update the magnetic field
        magnetic_field = h_field_update(
            electric_field,
            magnetic_field,
            id_field,
            mu_mapping,
            sigma_m_mapping,
            dt
        )

        # Update the magnetic field with PML
        for pml_layer in pml_layers:
            magnetic_field = pml_h_field_update(
                magnetic_field,
                pml_layer,
                id_field,
                mu_mapping,
                dt,
            )

        # Compute MUPS
        if step % 100 == 0:
            wp.synchronize()
            toc = time.time()
            mups = nr_cells * step / (toc - tic) / 1.0e6
            print(f"Iterations: {step}")
            print(f"MUPS: {mups}")