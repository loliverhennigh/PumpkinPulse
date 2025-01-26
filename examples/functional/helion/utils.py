# Utils

import os
import numpy as np
import warp as wp
import matplotlib.pyplot as plt
import dataclasses
import itertools
from tqdm import tqdm
from typing import List

wp.init()

from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.data import Fielduint8, Fieldfloat32
from pumpkin_pulse.operator import Operator
from pumpkin_pulse.operator.hydro import (
    PrimitiveToConservative,
    ConservativeToPrimitive,
    GetTimeStep,
    EulerUpdate,
)
from pumpkin_pulse.operator.mhd import (
    AddEMSourceTerms,
    GetCurrentDensity,
)
from pumpkin_pulse.operator.electromagnetism import (
    YeeElectricFieldUpdate,
    YeeMagneticFieldUpdate,
    InitializePML,
    PMLElectricFieldUpdate,
    PMLMagneticFieldUpdate,
    PMLPhiEUpdate,
    PMLPhiHUpdate,
)
from pumpkin_pulse.functional.indexing import periodic_indexing, periodic_setting
from pumpkin_pulse.functional.stencil import (
    p7_float32_stencil_type,
    p7_uint8_stencil_type,
    p4_float32_stencil_type,
    p4_uint8_stencil_type,
    faces_float32_type,
)
from pumpkin_pulse.operator.saver import FieldSaver

from geometry.reactor import Reactor


def capacitance_to_eps(
    capacitance: float,
    surface_area: float,
    thickness: float,
) -> float:
    return thickness * capacitance / surface_area

def switch_sigma_e(
    switch_sigma_e: float,
    switch_start_time: float,
    switch_time: float,
    switch_end_time: float,
) -> float:
    def _switch_sigma_e(t):
        if t < switch_start_time: # Before switch
            return 0.0
        elif t < switch_time + switch_start_time: # Ramp up
            alpha = (4.0 / switch_time)**2.0
            sigma_e = switch_sigma_e * np.exp(-alpha * (t - switch_time - switch_start_time)**2)
            return sigma_e
        elif t < switch_end_time: # Constant
            return switch_sigma_e
        elif t < switch_end_time + switch_time:
            alpha = (4.0 / switch_time)**2.0
            sigma_e = switch_sigma_e * (1.0 - np.exp(-alpha * (t - switch_end_time - switch_time)**2))
            return sigma_e
        else:
            return 0.0
    return _switch_sigma_e

class RampElectricField(Operator):

    @wp.kernel
    def _ramp_electric_field(
        electric_field: Fieldfloat32,
        id_field: Fielduint8,
        id_value: wp.uint8,
        value: wp.float32,
        direction: wp.int32,
    ):
        # Get index
        i, j, k = wp.tid()

        # Get id value
        local_id = id_field.data[0, i, j, k]

        # Check if id value is equal to the id value
        if id_value == local_id:

            # Set the electric field including edges
            if direction == 0:
                electric_field.data[0, i, j, k] = value
                electric_field.data[0, i, j, k + 1] = value
                electric_field.data[0, i, j + 1, k] = value
                electric_field.data[0, i, j + 1, k + 1] = value
            elif direction == 1:
                electric_field.data[1, i, j, k] = value
                electric_field.data[1, i + 1, j, k] = value
                electric_field.data[1, i, j, k + 1] = value
                electric_field.data[1, i + 1, j, k + 1] = value
            elif direction == 2:
                electric_field.data[2, i, j, k] = value
                electric_field.data[2, i, j + 1, k] = value
                electric_field.data[2, i + 1, j, k] = value
                electric_field.data[2, i + 1, j + 1, k] = value

    def __call__(
        self,
        electric_field: Fieldfloat32,
        id_field: Fielduint8,
        id_value: wp.uint8,
        value: wp.float32,
        direction: wp.int32,
    ):
        # Launch kernel
        wp.launch(
            self._ramp_electric_field,
            inputs=[
                electric_field,
                id_field,
                id_value,
                value,
                direction,
            ],
            dim=[s - 1 for s in electric_field.shape],
        )

        return electric_field

class MaterialIDMappings(Operator):

    def __call__(
        self,
        eps_mapping: dict,
        mu_mapping: dict,
        sigma_e_mapping: dict,
        sigma_h_mapping: dict,
        time: float,
    ):

        # Make numpy arrays
        np_eps_mapping = np.zeros(len(list(eps_mapping.keys())), dtype=np.float32)
        np_mu_mapping = np.zeros(len(list(mu_mapping.keys())), dtype=np.float32)
        np_sigma_e_mapping = np.zeros(len(list(sigma_e_mapping.keys())), dtype=np.float32)
        np_sigma_h_mapping = np.zeros(len(list(sigma_h_mapping.keys())), dtype=np.float32)
        for key in eps_mapping.keys():
            if callable(eps_mapping[key]):
                np_eps_mapping[key] = eps_mapping[key](time)
            else:
                np_eps_mapping[key] = eps_mapping[key]
        for key in mu_mapping.keys():
            if callable(mu_mapping[key]):
                np_mu_mapping[key] = mu_mapping[key](time)
            else:
                np_mu_mapping[key] = mu_mapping[key]
        for key in sigma_e_mapping.keys():
            if callable(sigma_e_mapping[key]):
                np_sigma_e_mapping[key] = sigma_e_mapping[key](time)
            else:
                np_sigma_e_mapping[key] = sigma_e_mapping[key]
        for key in sigma_h_mapping.keys():
            if callable(sigma_h_mapping[key]):
                np_sigma_h_mapping[key] = sigma_h_mapping[key](time)
            else:
                np_sigma_h_mapping[key] = sigma_h_mapping[key]

        # Convert to warp fields
        wp_eps_mapping = wp.from_numpy(np_eps_mapping, wp.float32)
        wp_mu_mapping = wp.from_numpy(np_mu_mapping, wp.float32)
        wp_sigma_e_mapping = wp.from_numpy(np_sigma_e_mapping, wp.float32)
        wp_sigma_h_mapping = wp.from_numpy(np_sigma_h_mapping, wp.float32)

        return wp_eps_mapping, wp_mu_mapping, wp_sigma_e_mapping, wp_sigma_h_mapping


class PlasmaInitializer(Operator):

    @wp.kernel
    def _initialize(
        density_i: Fieldfloat32,
        velocity_i: Fieldfloat32,
        pressure_i: Fieldfloat32,
        density_e: Fieldfloat32,
        velocity_e: Fieldfloat32,
        pressure_e: Fieldfloat32,
        id_field: Fielduint8,
        proton_mass: wp.float32,
        electron_mass: wp.float32,
        plasma_number_density: wp.float32,
        plasma_pressure: wp.float32,
    ):

        # Get index
        i, j, k = wp.tid()

        # Get index for id field
        id_field_i = i + density_i.offset[0] - id_field.offset[0]
        id_field_j = j + density_i.offset[1] - id_field.offset[1]
        id_field_k = k + density_i.offset[2] - id_field.offset[2]

        # Check if inside domain
        if id_field.data[0, id_field_i, id_field_j, id_field_k] != wp.uint8(0):
            return

        # Set density
        density_i.data[0, i, j, k] = plasma_number_density * proton_mass
        density_e.data[0, i, j, k] = plasma_number_density * electron_mass

        # Set pressure
        pressure_i.data[0, i, j, k] = plasma_pressure
        pressure_e.data[0, i, j, k] = plasma_pressure

        # Set velocity
        velocity_i.data[0, i, j, k] = 0.0
        velocity_i.data[1, i, j, k] = 0.0
        velocity_i.data[2, i, j, k] = 0.0
        velocity_e.data[0, i, j, k] = 0.0
        velocity_e.data[1, i, j, k] = 0.0
        velocity_e.data[2, i, j, k] = 0.0
            
    def __call__(
        self,
        density_i: Fieldfloat32,
        velocity_i: Fieldfloat32,
        pressure_i: Fieldfloat32,
        density_e: Fieldfloat32,
        velocity_e: Fieldfloat32,
        pressure_e: Fieldfloat32,
        id_field: Fielduint8,
        proton_mass: wp.float32,
        electron_mass: wp.float32,
        plasma_number_density: wp.float32,
        plasma_pressure: wp.float32
    ):
        # Launch kernel
        wp.launch(
            self._initialize,
            inputs=[
                density_i,
                velocity_i,
                pressure_i,
                density_e,
                velocity_e,
                pressure_e,
                id_field,
                proton_mass,
                electron_mass,
                plasma_number_density,
                plasma_pressure,
            ],
            dim=density_i.shape,
        )

        return density_i, velocity_i, pressure_i, density_e, velocity_e, pressure_e



def update_em_field(
    electric_field,
    magnetic_field,
    impressed_current,
    id_field,
    pml_layers,
    eps_mapping,
    mu_mapping,
    sigma_e_mapping,
    sigma_h_mapping,
    dt,
    pml_phi_e_update, 
    pml_phi_h_update,
    pml_e_field_update,
    pml_h_field_update,
    e_field_update,
    h_field_update,
):

    # Update the PML phi_e fields
    for pml_layer in pml_layers:
        pml_layer = pml_phi_e_update(
            magnetic_field,
            pml_layer,
        )

    # Update the electric field
    electric_field = e_field_update(
        electric_field,
        magnetic_field,
        impressed_current,
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
        sigma_h_mapping,
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

# Make boundary conditions functions
@wp.func
def apply_boundary_conditions_p7(
    rho_stencil: p7_float32_stencil_type,
    vx_stencil: p7_float32_stencil_type,
    vy_stencil: p7_float32_stencil_type,
    vz_stencil: p7_float32_stencil_type,
    p_stencil: p7_float32_stencil_type,
    id_stencil: p7_uint8_stencil_type,
):

    # Apply boundary conditions
    for i in range(6):

        # stencil index
        st_idx = i + 1

        # Wall
        if id_stencil[st_idx] == wp.uint8(1):

            # Get normal
            if st_idx == 1 or st_idx == 2:
                flip_vx = -1.0
            else:
                flip_vx = 1.0
            if st_idx == 3 or st_idx == 4:
                flip_vy = -1.0
            else:
                flip_vy = 1.0
            if st_idx == 5 or st_idx == 6:
                flip_vz = -1.0
            else:
                flip_vz = 1.0

            # Apply wall boundary condition
            rho_stencil[st_idx] = rho_stencil[0]
            vx_stencil[st_idx] = vx_stencil[0] * flip_vx
            vy_stencil[st_idx] = vy_stencil[0] * flip_vy
            vz_stencil[st_idx] = vz_stencil[0] * flip_vz
            p_stencil[st_idx] = p_stencil[0]

    return rho_stencil, vx_stencil, vy_stencil, vz_stencil, p_stencil

@wp.func
def apply_boundary_conditions_faces(
    rho_faces: faces_float32_type,
    vx_faces: faces_float32_type,
    vy_faces: faces_float32_type,
    vz_faces: faces_float32_type,
    p_faces: faces_float32_type,
    rho_stencil: p4_float32_stencil_type,
    vx_stencil: p4_float32_stencil_type,
    vy_stencil: p4_float32_stencil_type,
    vz_stencil: p4_float32_stencil_type,
    p_stencil: p4_float32_stencil_type,
    id_stencil: p4_uint8_stencil_type,
):

    # Apply boundary conditions
    if id_stencil[0] != wp.uint8(0):
        # RX face
        rho_faces[0] = rho_faces[1]
        vx_faces[0] = -vx_faces[1]
        vy_faces[0] = vy_faces[1]
        vz_faces[0] = vz_faces[1]
        p_faces[0] = p_faces[1]

        # RY face
        rho_faces[2] = rho_faces[3]
        vx_faces[2] = vx_faces[3]
        vy_faces[2] = -vy_faces[3]
        vz_faces[2] = vz_faces[3]
        p_faces[2] = p_faces[3]

        # RZ face
        rho_faces[4] = rho_faces[5]
        vx_faces[4] = vx_faces[5]
        vy_faces[4] = vy_faces[5]
        vz_faces[4] = -vz_faces[5]
        p_faces[4] = p_faces[5]

    # LX face
    if id_stencil[1] != wp.uint8(0):
        rho_faces[1] = rho_faces[0]
        vx_faces[1] = -vx_faces[0]
        vy_faces[1] = vy_faces[0]
        vz_faces[1] = vz_faces[0]
        p_faces[1] = p_faces[0]

    # LY face
    if id_stencil[2] != wp.uint8(0):
        rho_faces[3] = rho_faces[2]
        vx_faces[3] = vx_faces[2]
        vy_faces[3] = -vy_faces[2]
        vz_faces[3] = vz_faces[2]
        p_faces[3] = p_faces[2]

    # LZ face
    if id_stencil[3] != wp.uint8(0):
        rho_faces[5] = rho_faces[4]
        vx_faces[5] = vx_faces[4]
        vy_faces[5] = vy_faces[4]
        vz_faces[5] = -vz_faces[4]
        p_faces[5] = p_faces[4]

    return rho_faces, vx_faces, vy_faces, vz_faces, p_faces
