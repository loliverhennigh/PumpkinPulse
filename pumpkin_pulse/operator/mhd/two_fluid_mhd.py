# Base class for FV operators for two-fluid MHD equations

from typing import Union
import warp as wp

from pumpkin_pulse.data.field import Fieldfloat32, Fielduint8
from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.functional.indexing import periodic_indexing, periodic_atomic_add
from pumpkin_pulse.functional.finite_difference import centeral_difference
from pumpkin_pulse.functional.slope_limiter import minmod_slope_limiter
from pumpkin_pulse.functional.stencil import (
    p7_float32_stencil_type,
    p7_uint8_stencil_type,
    p4_float32_stencil_type,
    p4_uint8_stencil_type,
    p4_vec3f_stencil_type,
    faces_float32_type,
    get_p7_float32_stencil,
    get_p7_uint8_stencil,
    p4_stencil_to_faces,
)


class AddEMSourceTerms(Operator):
    """
    Add electromagnetic source terms to the euler equations
    """

    @wp.kernel
    def _add_em_source_terms(
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
        mass: Fieldfloat32,
        momentum: Fieldfloat32,
        energy: Fieldfloat32,
        id_field: Fielduint8,
        species_mass: wp.float32,
        charge: wp.float32,
        mu_0: wp.float32,
        dt: wp.float32,
    ):
        # Get index
        i, j, k = wp.tid()

        # Get em index
        em_i = i + density.offset[0] - electric_field.offset[0]
        em_j = j + density.offset[1] - electric_field.offset[1]
        em_k = k + density.offset[2] - electric_field.offset[2]

        # Get id index
        id_i = i + density.offset[0] - id_field.offset[0]
        id_j = j + density.offset[1] - id_field.offset[1]
        id_k = k + density.offset[2] - id_field.offset[2]

        # Check if cell is fluid
        if id_field.data[0, id_i, id_j, id_k] != wp.uint8(0):
            return

        # Get volume
        volume = density.spacing[0] * density.spacing[1] * density.spacing[2]

        # Get primitive variables
        rho = density.data[0, i, j, k]
        vel = wp.vec3f(
            velocity.data[0, i, j, k],
            velocity.data[1, i, j, k],
            velocity.data[2, i, j, k],
        )

        # Get electric and magnetic fields
        ex_0_0_0 = periodic_indexing(electric_field.data, electric_field.shape, 0, em_i, em_j, em_k)
        ex_0_1_0 = periodic_indexing(electric_field.data, electric_field.shape, 0, em_i, em_j + 1, em_k)
        ex_0_0_1 = periodic_indexing(electric_field.data, electric_field.shape, 0, em_i, em_j, em_k + 1)
        ex_0_1_1 = periodic_indexing(electric_field.data, electric_field.shape, 0, em_i, em_j + 1, em_k + 1)
        ey_0_0_0 = periodic_indexing(electric_field.data, electric_field.shape, 1, em_i, em_j, em_k)
        ey_1_0_0 = periodic_indexing(electric_field.data, electric_field.shape, 1, em_i + 1, em_j, em_k)
        ey_0_0_1 = periodic_indexing(electric_field.data, electric_field.shape, 1, em_i, em_j, em_k + 1)
        ey_1_0_1 = periodic_indexing(electric_field.data, electric_field.shape, 1, em_i + 1, em_j, em_k + 1)
        ez_0_0_0 = periodic_indexing(electric_field.data, electric_field.shape, 2, em_i, em_j, em_k)
        ez_1_0_0 = periodic_indexing(electric_field.data, electric_field.shape, 2, em_i + 1, em_j, em_k)
        ez_0_1_0 = periodic_indexing(electric_field.data, electric_field.shape, 2, em_i, em_j + 1, em_k)
        ez_1_1_0 = periodic_indexing(electric_field.data, electric_field.shape, 2, em_i + 1, em_j + 1, em_k)
        hx_0_0_0 = periodic_indexing(magnetic_field.data, magnetic_field.shape, 0, em_i, em_j, em_k)
        hx_1_0_0 = periodic_indexing(magnetic_field.data, magnetic_field.shape, 0, em_i + 1, em_j, em_k)
        hy_0_0_0 = periodic_indexing(magnetic_field.data, magnetic_field.shape, 1, em_i, em_j, em_k)
        hy_0_1_0 = periodic_indexing(magnetic_field.data, magnetic_field.shape, 1, em_i, em_j + 1, em_k)
        hz_0_0_0 = periodic_indexing(magnetic_field.data, magnetic_field.shape, 2, em_i, em_j, em_k)
        hz_0_0_1 = periodic_indexing(magnetic_field.data, magnetic_field.shape, 2, em_i, em_j, em_k + 1)

        # Get cell centered averages
        e = wp.vec3f(
            0.25 * (ex_0_0_0 + ex_0_1_0 + ex_0_0_1 + ex_0_1_1),
            0.25 * (ey_0_0_0 + ey_1_0_0 + ey_0_0_1 + ey_1_0_1),
            0.25 * (ez_0_0_0 + ez_1_0_0 + ez_0_1_0 + ez_1_1_0),
        )
        h = wp.vec3f(
            0.5 * (hx_0_0_0 + hx_1_0_0),
            0.5 * (hy_0_0_0 + hy_0_1_0),
            0.5 * (hz_0_0_0 + hz_0_0_1),
        )
        b = mu_0 * h

        # Get vel cross b
        vel_cross_b = wp.cross(vel, b)

        # Get density mass charge ratio
        density_mass_charge = rho * charge / species_mass

        # Add source terms
        # Momentum
        momentum.data[0, i, j, k] += dt * volume * density_mass_charge * (e[0] + vel_cross_b[0])
        momentum.data[1, i, j, k] += dt * volume * density_mass_charge * (e[1] + vel_cross_b[1])
        momentum.data[2, i, j, k] += dt * volume * density_mass_charge * (e[2] + vel_cross_b[2])

        # Energy
        energy.data[0, i, j, k] += dt * volume * density_mass_charge * wp.dot(vel, e)

    def __call__(
        self,
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
        mass: Fieldfloat32,
        momentum: Fieldfloat32,
        energy: Fieldfloat32,
        id_field: Fielduint8,
        species_mass: float,
        charge: float,
        mu_0: float,
        dt: float,
    ):
        # Launch kernel
        wp.launch(
            self._add_em_source_terms,
            inputs=[
                density,
                velocity,
                pressure,
                electric_field,
                magnetic_field,
                mass,
                momentum,
                energy,
                id_field,
                species_mass,
                charge,
                mu_0,
                dt,
            ],
            dim=density.shape,
        )

        return mass, momentum, energy


class GetCurrentDensity(Operator):

    @wp.kernel
    def _get_current_density(
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
        current_density: Fieldfloat32,
        field_id: Fielduint8,
        species_mass: wp.float32,
        charge: wp.float32,
    ):

        # Get index
        i, j, k = wp.tid()

        ## Get cell id
        #id_1_0_0 = field_id.data[0, i, j - 1, k - 1]
        #id_0_1_0 = field_id.data[0, i - 1, j, k - 1]
        #id_1_1_0 = field_id.data[0, i, j, k - 1]
        #id_0_0_1 = field_id.data[0, i - 1, j - 1, k]
        #id_1_0_1 = field_id.data[0, i, j - 1, k]
        #id_0_1_1 = field_id.data[0, i - 1, j, k]
        #id_1_1_1 = field_id.data[0, i, j, k]

        ## Check if cell is fluid
        #if (id_1_0_0 != wp.uint8(0) or
        #    id_0_1_0 != wp.uint8(0) or
        #    id_1_1_0 != wp.uint8(0) or
        #    id_0_0_1 != wp.uint8(0) or
        #    id_1_0_1 != wp.uint8(0) or
        #    id_0_1_1 != wp.uint8(0) or
        #    id_1_1_1 != wp.uint8(0)):
        #    return

        # Get primitive variables
        rho_1_0_0 = periodic_indexing(density.data, density.shape, 0, i, j - 1, k - 1)
        rho_0_1_0 = periodic_indexing(density.data, density.shape, 0, i - 1, j, k - 1)
        rho_1_1_0 = periodic_indexing(density.data, density.shape, 0, i, j, k - 1)
        rho_0_0_1 = periodic_indexing(density.data, density.shape, 0, i - 1, j - 1, k)
        rho_1_0_1 = periodic_indexing(density.data, density.shape, 0, i, j - 1, k)
        rho_0_1_1 = periodic_indexing(density.data, density.shape, 0, i - 1, j, k)
        rho_1_1_1 = periodic_indexing(density.data, density.shape, 0, i, j, k)
        vx_1_0_0 = periodic_indexing(velocity.data, velocity.shape, 0, i, j - 1, k - 1)
        vx_1_1_0 = periodic_indexing(velocity.data, velocity.shape, 0, i, j, k - 1)
        vx_1_0_1 = periodic_indexing(velocity.data, velocity.shape, 0, i, j - 1, k)
        vx_1_1_1 = periodic_indexing(velocity.data, velocity.shape, 0, i, j, k)
        vy_0_1_0 = periodic_indexing(velocity.data, velocity.shape, 1, i - 1, j, k - 1)
        vy_1_1_0 = periodic_indexing(velocity.data, velocity.shape, 1, i, j, k - 1)
        vy_0_1_1 = periodic_indexing(velocity.data, velocity.shape, 1, i - 1, j, k)
        vy_1_1_1 = periodic_indexing(velocity.data, velocity.shape, 1, i, j, k)
        vz_0_0_1 = periodic_indexing(velocity.data, velocity.shape, 2, i - 1, j - 1, k)
        vz_1_0_1 = periodic_indexing(velocity.data, velocity.shape, 2, i, j - 1, k)
        vz_0_1_1 = periodic_indexing(velocity.data, velocity.shape, 2, i - 1, j, k)
        vz_1_1_1 = periodic_indexing(velocity.data, velocity.shape, 2, i, j, k)

        # Get electric field averaged values
        rho_jx = (rho_1_1_1 + rho_1_0_1 + rho_1_1_0 + rho_1_0_0) / 4.0
        rho_jy = (rho_1_1_1 + rho_0_1_1 + rho_1_1_0 + rho_0_1_0) / 4.0
        rho_jz = (rho_1_1_1 + rho_0_1_1 + rho_1_0_1 + rho_0_0_1) / 4.0
        vx_jx = (vx_1_1_1 + vx_1_0_1 + vx_1_1_0 + vx_1_0_0) / 4.0 
        vy_jy = (vy_1_1_1 + vy_0_1_1 + vy_1_1_0 + vy_0_1_0) / 4.0
        vz_jz = (vz_1_1_1 + vz_0_1_1 + vz_1_0_1 + vz_0_0_1) / 4.0

        # Get mass charge ratio
        mass_charge = charge / species_mass

        # Add current density
        current_density.data[0, i, j, k] += mass_charge * rho_jx * vx_jx
        current_density.data[1, i, j, k] += mass_charge * rho_jy * vy_jy
        current_density.data[2, i, j, k] += mass_charge * rho_jz * vz_jz

    def __call__(
        self,
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
        current_density: Fieldfloat32,
        field_id: Fielduint8,
        species_mass: float,
        charge: float,
    ):
        # Launch kernel
        wp.launch(
            self._get_current_density,
            inputs=[density, velocity, pressure, current_density, field_id, species_mass, charge],
            dim=density.shape,
        )

        return current_density
