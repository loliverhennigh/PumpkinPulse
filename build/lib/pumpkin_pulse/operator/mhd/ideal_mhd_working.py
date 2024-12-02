# Ideal MHD operators

from typing import Union
import warp as wp

from pumpkin_pulse.data.field import Fieldfloat32, Fielduint8
from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.functional.indexing import periodic_indexing, periodic_setting
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

class PrimitiveToConservative(Operator):
    """
    Convert primitive variables to conservative variables
    """

    @wp.kernel
    def _primitive_to_conservative(
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
        cell_magnetic_field: Fieldfloat32,
        mass: Fieldfloat32,
        momentum: Fieldfloat32,
        energy: Fieldfloat32,
        gamma: wp.float32,
    ):
        # Get index
        i, j, k = wp.tid()

        # Get volume
        volume = density.spacing[0] * density.spacing[1] * density.spacing[2]

        # Get variables
        rho = density.data[0, i, j, k]
        vel = wp.vec3f(
            velocity.data[0, i, j, k],
            velocity.data[1, i, j, k],
            velocity.data[2, i, j, k],
        )
        p = pressure.data[0, i, j, k]
        cell_b = wp.vec3f(
            cell_magnetic_field.data[0, i, j, k],
            cell_magnetic_field.data[1, i, j, k],
            cell_magnetic_field.data[2, i, j, k],
        )

        # Set mass
        mass.data[0, i, j, k] = rho * volume

        # Set momentum
        momentum.data[0, i, j, k] = rho * vel[0] * volume
        momentum.data[1, i, j, k] = rho * vel[1] * volume
        momentum.data[2, i, j, k] = rho * vel[2] * volume

        # Set energy
        energy.data[0, i, j, k] = (
            (p - 0.5 * wp.dot(cell_b, cell_b)) / (gamma - 1.0)
            + 0.5 * rho * wp.dot(vel, vel)
            + 0.5 * wp.dot(cell_b, cell_b)
        ) * volume

    def __call__(
        self,
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
        cell_magnetic_field: Fieldfloat32,
        mass: Fieldfloat32,
        momentum: Fieldfloat32,
        energy: Fieldfloat32,
        gamma: float,
    ):
        # Launch kernel
        wp.launch(
            self._primitive_to_conservative,
            inputs=[
                density,
                velocity,
                pressure,
                cell_magnetic_field,
                mass,
                momentum,
                energy,
                gamma,
            ],
            dim=density.shape,
        )

        return mass, momentum, energy


class ConservativeToPrimitive(Operator):
    """
    Convert conservative variables to primitive variables
    """

    @wp.kernel
    def _conservative_to_primitive(
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
        cell_magnetic_field: Fieldfloat32,
        mass: Fieldfloat32,
        momentum: Fieldfloat32,
        energy: Fieldfloat32,
        gamma: wp.float32,
    ):
        # Get index
        i, j, k = wp.tid()

        # Get volume
        volume = density.spacing[0] * density.spacing[1] * density.spacing[2]

        # Get variables
        m = mass.data[0, i, j, k]
        mom = wp.vec3f(
            momentum.data[0, i, j, k],
            momentum.data[1, i, j, k],
            momentum.data[2, i, j, k],
        )
        e = energy.data[0, i, j, k]
        cell_b = wp.vec3f(
            cell_magnetic_field.data[0, i, j, k],
            cell_magnetic_field.data[1, i, j, k],
            cell_magnetic_field.data[2, i, j, k],
        )

        # Get primitive variables
        rho = m / volume
        vel = mom / rho / volume

        # Set density
        density.data[0, i, j, k] = rho

        # Set velocity
        velocity.data[0, i, j, k] = vel[0]
        velocity.data[1, i, j, k] = vel[1]
        velocity.data[2, i, j, k] = vel[2]

        # Set pressure
        pressure.data[0, i, j, k] = (
            (e / volume) - 0.5 * rho * wp.dot(vel, vel) - 0.5 * wp.dot(cell_b, cell_b)
        ) * (gamma - 1.0) + 0.5 * wp.dot(cell_b, cell_b)

    def __call__(
        self,
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
        cell_magnetic_field: Fieldfloat32,
        mass: Fieldfloat32,
        momentum: Fieldfloat32,
        energy: Fieldfloat32,
        gamma: float,
    ):
        # Launch kernel
        wp.launch(
            self._conservative_to_primitive,
            inputs=[
                density,
                velocity,
                pressure,
                cell_magnetic_field,
                mass,
                momentum,
                energy,
                gamma,
            ],
            dim=density.shape,
        )

        return density, velocity, pressure


class GetTimeStep(Operator):
    """
    Get time step based on CFL condition for ideal MHD equations
    """

    @wp.kernel
    def _get_time_step(
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
        cell_magnetic_field: Fieldfloat32,
        id_field: Fielduint8,
        time_step: wp.array(dtype=wp.float32),
        courant_factor: wp.float32,
        gamma: wp.float32,
    ):
        # Get index
        i, j, k = wp.tid()

        # Check if inside domain
        if id_field.data[0, i, j, k] != wp.uint8(0):
            return

        # Get min spacing
        min_spacing = wp.min(
            wp.min(density.spacing[0], density.spacing[1]), density.spacing[2]
        )

        # Get variables
        rho = density.data[0, i, j, k]
        vel = wp.vec3f(
            velocity.data[0, i, j, k],
            velocity.data[1, i, j, k],
            velocity.data[2, i, j, k],
        )
        p = pressure.data[0, i, j, k]
        cell_b = wp.vec3f(
            cell_magnetic_field.data[0, i, j, k],
            cell_magnetic_field.data[1, i, j, k],
            cell_magnetic_field.data[2, i, j, k],
        )

        # Get CFL
        c0 = wp.sqrt(gamma * (p - 0.5 * wp.dot(cell_b, cell_b)) / rho + 1e-12)
        ca = wp.sqrt(wp.dot(cell_b, cell_b) / rho + 1e-12)
        cf = wp.sqrt(
            0.5 * (c0**2.0 + ca**2.0)
            + 0.5 * wp.sqrt((c0**2.0 + ca**2.0) ** 2.0)
            + 1e-12
        )
        dt = courant_factor * min_spacing / (cf + wp.length(vel))

        # Chect dt
        if dt > 1e4:
            print("Time step too large")
        if wp.isnan(dt):
            print("Time step is NaN")

        # Update time step
        wp.atomic_min(time_step, 0, dt)

    def __call__(
        self,
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
        cell_magnetic_field: Fieldfloat32,
        id_field: Fielduint8,
        courant_factor: float,
        gamma: float,
    ):
        # Set time step
        time_step = wp.array([1e10], dtype=wp.float32)

        # Launch kernel
        wp.launch(
            self._get_time_step,
            inputs=[
                density,
                velocity,
                pressure,
                cell_magnetic_field,
                id_field,
                time_step,
                courant_factor,
                gamma,
            ],
            dim=density.shape,
        )

        return time_step.numpy()[0]


class IdealMHDUpdate(Operator):
    """
    Update conservative variables using ideal MHD equations
    """

    @wp.func
    def _extrapolate_half_time_step(
        rho: wp.float32,
        rho_dxyz: wp.vec3f,
        vx: wp.float32,
        vx_dxyz: wp.vec3f,
        vy: wp.float32,
        vy_dxyz: wp.vec3f,
        vz: wp.float32,
        vz_dxyz: wp.vec3f,
        p: wp.float32,
        p_dxyz: wp.vec3f,
        cell_bx: wp.float32,
        cell_bx_dxyz: wp.vec3f,
        cell_by: wp.float32,
        cell_by_dxyz: wp.vec3f,
        cell_bz: wp.float32,
        cell_bz_dxyz: wp.vec3f,
        gamma: wp.float32,
        dt: wp.float32,
    ):

        # Needed values
        dt_half = 0.5 * dt
        v = wp.vec3f(vx, vy, vz)
        cell_b = wp.vec3f(cell_bx, cell_by, cell_bz)
        v_dxyz_trace = vx_dxyz[0] + vy_dxyz[1] + vz_dxyz[2]
        v_dot_rho_dxyz = wp.dot(v, rho_dxyz)
        v_dx_dot_v = wp.dot(v, vx_dxyz)
        v_dy_dot_v = wp.dot(v, vy_dxyz)
        v_dz_dot_v = wp.dot(v, vz_dxyz)
        v_dot_p_dxyz = wp.dot(v, p_dxyz)
        cell_b_dxyz_trace = cell_bx_dxyz[0] + cell_by_dxyz[1] + cell_bz_dxyz[2]
        v_dot_cell_b = wp.dot(v, cell_b)

        # Extrapolate values
        # Extrapolate rho half-step in time
        rho_prime = rho - dt_half * (
            v_dot_rho_dxyz
            + rho * v_dxyz_trace
        )

        # Extrapolate velocity half-step in time
        vx_prime = vx - dt_half * (
            v_dx_dot_v
            + p_dxyz[0] / rho
            - (cell_bx / rho) * (cell_b_dxyz_trace + cell_bx_dxyz[0])
            - (cell_by / rho) * cell_bx_dxyz[1]
            - (cell_bz / rho) * cell_bx_dxyz[2]
        )
        vy_prime = vy - dt_half * (
            v_dy_dot_v  
            + p_dxyz[1] / rho
            - (cell_bx / rho) * cell_by_dxyz[0]
            - (cell_by / rho) * (cell_b_dxyz_trace + cell_by_dxyz[1])
            - (cell_bz / rho) * cell_by_dxyz[2]
        )
        vz_prime = vz - dt_half * (
            v_dz_dot_v
            + p_dxyz[2] / rho
            - (cell_bx / rho) * cell_bz_dxyz[0]
            - (cell_by / rho) * cell_bz_dxyz[1]
            - (cell_bz / rho) * (cell_b_dxyz_trace + cell_bz_dxyz[2])
        )

        # Extrapolate pressure half-step in time
        p_prime = p - dt_half * (
            (gamma * (p - 0.5 * wp.dot(cell_b, cell_b)) + cell_by ** 2.0 + cell_bz ** 2.0) * vx_dxyz[0] # dvx/dx term
            - cell_bx * cell_by * vy_dxyz[0] # dvy/dx term
            - cell_bx * cell_bz * vz_dxyz[0] # dvz/dx term
            + vx * p_dxyz[0] # dp/dx term
            + (gamma - 2.0) * v_dot_cell_b * cell_bx_dxyz[0] # dBx/dx term
            - cell_bx * cell_by * vx_dxyz[1] # dvx/dy term
            + (gamma * (p - 0.5 * wp.dot(cell_b, cell_b)) + cell_bx ** 2.0 + cell_bz ** 2.0) * vy_dxyz[1] # dvy/dy term
            - cell_by * cell_bz * vz_dxyz[1] # dvz/dy term
            + vy * p_dxyz[1] # dp/dy term
            + (gamma - 2.0) * v_dot_cell_b * cell_by_dxyz[1] # dBy/dy term
            - cell_bx * cell_bz * vx_dxyz[2] # dvx/dz term
            - cell_by * cell_bz * vy_dxyz[2] # dvy/dz term
            + (gamma * (p - 0.5 * wp.dot(cell_b, cell_b)) + cell_bx ** 2.0 + cell_by ** 2.0) * vz_dxyz[2] # dvz/dz term
            + vz * p_dxyz[2] # dp/dz term
            + (gamma - 2.0) * v_dot_cell_b * cell_bz_dxyz[2] # dBz/dz term
        )

        # Extrapolate magnetic field half-step in time
        cell_bx_prime = cell_bx - dt_half * (
            cell_bx * vy_dxyz[1]
            + cell_bx * vz_dxyz[2]
            - cell_by * vx_dxyz[1]
            - cell_bz * vx_dxyz[2]
            - vx * cell_by_dxyz[1]
            - vx * cell_bz_dxyz[2]
            + vy * cell_bx_dxyz[1]
            + vz * cell_bx_dxyz[2]
        )
        cell_by_prime = cell_by - dt_half * (
            -cell_bx * vy_dxyz[0]
            + cell_by * vx_dxyz[0]
            + cell_by * vz_dxyz[2]
            - cell_bz * vy_dxyz[2]
            + vx * cell_by_dxyz[0]
            - vy * cell_bx_dxyz[0]
            - vy * cell_bz_dxyz[2]
            + vz * cell_by_dxyz[2]
        )
        cell_bz_prime = cell_bz - dt_half * (
            -cell_bx * vz_dxyz[0]
            - cell_by * vz_dxyz[1]
            + cell_bz * vx_dxyz[0]
            + cell_bz * vy_dxyz[1]
            + vx * cell_bz_dxyz[0]
            + vy * cell_bz_dxyz[1]
            - vz * cell_bx_dxyz[0]
            - vz * cell_by_dxyz[1]
        )

        return rho_prime, vx_prime, vy_prime, vz_prime, p_prime, cell_bx_prime, cell_by_prime, cell_bz_prime

    @wp.func
    def _compute_fluxes(
        rho_l: wp.float32,
        rho_r: wp.float32,
        vx_l: wp.float32,
        vx_r: wp.float32,
        vy_l: wp.float32,
        vy_r: wp.float32,
        vz_l: wp.float32,
        vz_r: wp.float32,
        p_l: wp.float32,
        p_r: wp.float32,
        cell_bx_l: wp.float32,
        cell_bx_r: wp.float32,
        cell_by_l: wp.float32,
        cell_by_r: wp.float32,
        cell_bz_l: wp.float32,
        cell_bz_r: wp.float32,
        gamma: wp.float32,
    ):

        # Get needed quantities
        v_l = wp.vec3f(vx_l, vy_l, vz_l)
        v_r = wp.vec3f(vx_r, vy_r, vz_r)
        cell_b_l = wp.vec3f(cell_bx_l, cell_by_l, cell_bz_l)
        cell_b_r = wp.vec3f(cell_bx_r, cell_by_r, cell_bz_r)

        # Compute energies
        e_l = (
            (p_l - 0.5 * wp.dot(cell_b_l, cell_b_l)) / (gamma - 1.0)
            + 0.5 * rho_l * wp.dot(v_l, v_l)
            + 0.5 * wp.dot(cell_b_l, cell_b_l)
        )
        e_r = (
            (p_r - 0.5 * wp.dot(cell_b_r, cell_b_r)) / (gamma - 1.0)
            + 0.5 * rho_r * wp.dot(v_r, v_r)
            + 0.5 * wp.dot(cell_b_r, cell_b_r)
        )

        # Compute averages
        rho_avg = 0.5 * (rho_l + rho_r)
        mom_x_avg = 0.5 * (rho_l * vx_l + rho_r * vx_r)
        mom_y_avg = 0.5 * (rho_l * vy_l + rho_r * vy_r)
        mom_z_avg = 0.5 * (rho_l * vz_l + rho_r * vz_r)
        e_avg = 0.5 * (e_l + e_r)
        cell_bx_avg = 0.5 * (cell_bx_l + cell_bx_r)
        cell_by_avg = 0.5 * (cell_by_l + cell_by_r)
        cell_bz_avg = 0.5 * (cell_bz_l + cell_bz_r)
        p_avg = (gamma - 1.0) * (
            e_avg
            - 0.5 * (mom_x_avg**2.0 + mom_y_avg**2.0 + mom_z_avg**2.0) / rho_avg
            - 0.5 * (cell_bx_avg**2.0 + cell_by_avg**2.0 + cell_bz_avg**2.0)
        ) + 0.5 * (cell_bx_avg**2.0 + cell_by_avg**2.0 + cell_bz_avg**2.0)
            
        # Compute fluxes
        flux_mass = mom_x_avg
        flux_mom_x = mom_x_avg**2.0 / rho_avg + p_avg - cell_bx_avg**2.0
        flux_mom_y = mom_x_avg * mom_y_avg / rho_avg - cell_bx_avg * cell_by_avg
        flux_mom_z = mom_x_avg * mom_z_avg / rho_avg - cell_bx_avg * cell_bz_avg
        flux_energy = (
            mom_x_avg * (e_avg + p_avg) / rho_avg 
            - cell_bx_avg * (
                cell_bx_avg * mom_x_avg
                + cell_by_avg * mom_y_avg
                + cell_bz_avg * mom_z_avg
            ) / rho_avg
        )
        flux_by = (cell_by_avg * mom_x_avg - cell_bx_avg * mom_y_avg) / rho_avg
        flux_bz = (cell_bz_avg * mom_x_avg - cell_bx_avg * mom_z_avg) / rho_avg

        # Compute wave speeds
        c0_out = wp.sqrt(gamma * (
            p_out
            - 0.5 * wp.dot(cell_b_out, cell_b_out)
        ) / rho_out)
        c0_in = wp.sqrt(gamma * (
            p_in
            - 0.5 * wp.dot(cell_b_in, cell_b_in)
        ) / rho_in)
        ca_out = wp.sqrt(wp.dot(cell_b_out, cell_b_out) / rho_out)
        ca_in = wp.sqrt(wp.dot(cell_b_in, cell_b_in) / rho_in)
        cf_out = wp.sqrt(
            0.5 * (c0_out**2.0 + ca_out**2.0)
            + 0.5 * wp.sqrt((c0_out**2.0 + ca_out**2.0) ** 2.0)
        )
        cf_in = wp.sqrt(
            0.5 * (c0_in**2.0 + ca_in**2.0)
            + 0.5 * wp.sqrt((c0_in**2.0 + ca_in**2.0) ** 2.0)
        )
        if dim == 0:
            c_out = cf_out + wp.abs(vx_out)
            c_in = cf_in + wp.abs(vx_in)
        elif dim == 1:
            c_out = cf_out + wp.abs(vy_out)
            c_in = cf_in + wp.abs(vy_in)
        elif dim == 2:
            c_out = cf_out + wp.abs(vz_out)
            c_in = cf_in + wp.abs(vz_in)
        c = wp.max(c_out, c_in)

        # Stabilizing diffusion term
        c_flux_mass = c * 0.5 * (rho_in - rho_out)
        c_flux_mom_x = c * 0.5 * (rho_in * vx_in - rho_out * vx_out)
        c_flux_mom_y = c * 0.5 * (rho_in * vy_in - rho_out * vy_out)
        c_flux_mom_z = c * 0.5 * (rho_in * vz_in - rho_out * vz_out)
        c_flux_energy = c * 0.5 * (e_in - e_out)
        c_flux_bx = c * 0.5 * (cell_bx_in - cell_bx_out)
        c_flux_by = c * 0.5 * (cell_by_in - cell_by_out)
        c_flux_bz = c * 0.5 * (cell_bz_in - cell_bz_out)

        # Compute fluxes
        flux_mass   = - sign * llf_flux_mass   - c_flux_mass
        flux_mom_x  = - sign * llf_flux_mom_x  - c_flux_mom_x
        flux_mom_y  = - sign * llf_flux_mom_y  - c_flux_mom_y
        flux_mom_z  = - sign * llf_flux_mom_z  - c_flux_mom_z
        flux_energy = - sign * llf_flux_energy - c_flux_energy
        flux_bx = sign * llf_flux_bx + c_flux_bx
        flux_by = sign * llf_flux_by + c_flux_by
        flux_bz = sign * llf_flux_bz + c_flux_bz

        return flux_mass, flux_mom_x, flux_mom_y, flux_mom_z, flux_energy, flux_bx, flux_by, flux_bz

    @wp.func
    def _do_nothing_boundary_conditions_p7(
        rho_stencil: p7_float32_stencil_type,
        vx_stencil: p7_float32_stencil_type,
        vy_stencil: p7_float32_stencil_type,
        vz_stencil: p7_float32_stencil_type,
        p_stencil: p7_float32_stencil_type,
        cell_bx_stencil: p7_float32_stencil_type,
        cell_by_stencil: p7_float32_stencil_type,
        cell_bz_stencil: p7_float32_stencil_type,
        id_stencil: p7_uint8_stencil_type,
    ):
        return (
            rho_stencil,
            vx_stencil,
            vy_stencil,
            vz_stencil,
            p_stencil,
            cell_bx_stencil,
            cell_by_stencil,
            cell_bz_stencil,
        )

    @wp.func
    def _do_nothing_boundary_conditions_faces(
        rho_faces: faces_float32_type,
        vx_faces: faces_float32_type,
        vy_faces: faces_float32_type,
        vz_faces: faces_float32_type,
        p_faces: faces_float32_type,
        cell_bx_faces: faces_float32_type,
        cell_by_faces: faces_float32_type,
        cell_bz_faces: faces_float32_type,
        rho_stencil: p4_float32_stencil_type,
        vx_stencil: p4_float32_stencil_type,
        vy_stencil: p4_float32_stencil_type,
        vz_stencil: p4_float32_stencil_type,
        p_stencil: p4_float32_stencil_type,
        cell_bx_stencil: p4_float32_stencil_type,
        cell_by_stencil: p4_float32_stencil_type,
        cell_bz_stencil: p4_float32_stencil_type,
        id_stencil: p4_uint8_stencil_type,
    ):
        return (
            rho_faces,
            vx_faces,
            vy_faces,
            vz_faces,
            p_faces,
            cell_bx_faces,
            cell_by_faces,
            cell_bz_faces,
        )


    def __init__(
        self,
        apply_boundary_conditions_p7: callable = None,
        apply_boundary_conditions_faces: callable = None,
        slope_limiter: callable = None,
    ):
        # Set boundary conditions functions
        if apply_boundary_conditions_p7 is None:
            apply_boundary_conditions_p7 = self._do_nothing_boundary_conditions_p7
        if apply_boundary_conditions_faces is None:
            apply_boundary_conditions_faces = self._do_nothing_boundary_conditions_faces

        # Set slope limiter
        if slope_limiter is None:
            slope_limiter = minmod_slope_limiter

        # Make 3d slope limiter function
        @wp.func
        def slope_limiter_3d(
            v_1_1_1: wp.float32,
            v_0_1_1: wp.float32,
            v_2_1_1: wp.float32,
            v_1_0_1: wp.float32,
            v_1_2_1: wp.float32,
            v_1_1_0: wp.float32,
            v_1_1_2: wp.float32,
            v_dxyz: wp.vec3f,
            spacing: wp.vec3f,
        ):
            v_dx = slope_limiter(v_0_1_1, v_1_1_1, v_2_1_1, v_dxyz[0], spacing[0], 1e-8)
            v_dy = slope_limiter(v_1_0_1, v_1_1_1, v_1_2_1, v_dxyz[1], spacing[1], 1e-8)
            v_dz = slope_limiter(v_1_1_0, v_1_1_1, v_1_1_2, v_dxyz[2], spacing[2], 1e-8)
            return wp.vec3f(v_dx, v_dy, v_dz)

        # Make derivative function
        @wp.func
        def p7_stencil_to_dxyz(
            stencil: p7_float32_stencil_type,
            spacing: wp.vec3f,
        ):
            # Compute derivatives
            v_dxyz = centeral_difference(
                stencil[1], stencil[2], stencil[3], stencil[4], stencil[5], stencil[6], spacing
            )
    
            # Slope limiter
            v_dxyz = slope_limiter_3d(
                stencil[0],
                stencil[1],
                stencil[2],
                stencil[3],
                stencil[4],
                stencil[5],
                stencil[6],
                v_dxyz,
                spacing,
            )
    
            return v_dxyz


        # Generate kernel
        @wp.kernel
        def _ideal_mhd_update(
            density: Fieldfloat32,
            velocity: Fieldfloat32,
            pressure: Fieldfloat32,
            cell_magnetic_field: Fieldfloat32,
            mass: Fieldfloat32,
            momentum: Fieldfloat32,
            energy: Fieldfloat32,
            flux_magnetic_field: Fieldfloat32,
            id_field: Fielduint8,
            gamma: wp.float32,
            dt: wp.float32,
        ):
            # Get index
            i, j, k = wp.tid()

            ## Check if inside domain
            #if id_field.data[0, i, j, k] != wp.uint8(0):
            #    return

            # Make 4p stencil
            rho_p4_stencil = p4_float32_stencil_type()
            vx_p4_stencil = p4_float32_stencil_type()
            vy_p4_stencil = p4_float32_stencil_type()
            vz_p4_stencil = p4_float32_stencil_type()
            p_p4_stencil = p4_float32_stencil_type()
            cell_bx_4p_stencil = p4_float32_stencil_type()
            cell_by_4p_stencil = p4_float32_stencil_type()
            cell_bz_4p_stencil = p4_float32_stencil_type()
            id_p4_stencil = p4_uint8_stencil_type()

            # Make p7 stencil derivatives
            rho_p4_stencil_dxyz = p4_vec3f_stencil_type()
            vx_p4_stencil_dxyz = p4_vec3f_stencil_type()
            vy_p4_stencil_dxyz = p4_vec3f_stencil_type()
            vz_p4_stencil_dxyz = p4_vec3f_stencil_type()
            p_p4_stencil_dxyz = p4_vec3f_stencil_type()
            cell_bx_p4_stencil_dxyz = p4_vec3f_stencil_type()
            cell_by_p4_stencil_dxyz = p4_vec3f_stencil_type()
            cell_bz_p4_stencil_dxyz = p4_vec3f_stencil_type()

            # Fill 4p stencil
            for c in range(4):
                # Get index offset
                if c == 0:
                    i_offset, j_offset, k_offset = 0, 0, 0
                elif c == 1:
                    i_offset, j_offset, k_offset = 1, 0, 0
                elif c == 2:
                    i_offset, j_offset, k_offset = 0, 1, 0
                elif c == 3:
                    i_offset, j_offset, k_offset = 0, 0, 1

                # Get p7 stencil
                rho_p7_stencil = get_p7_float32_stencil(
                    density.data, density.shape, 0, i + i_offset, j + j_offset, k + k_offset
                )
                vx_p7_stencil = get_p7_float32_stencil(
                    velocity.data, density.shape, 0, i + i_offset, j + j_offset, k + k_offset
                )
                vy_p7_stencil = get_p7_float32_stencil(
                    velocity.data, density.shape, 1, i + i_offset, j + j_offset, k + k_offset
                )
                vz_p7_stencil = get_p7_float32_stencil(
                    velocity.data, density.shape, 2, i + i_offset, j + j_offset, k + k_offset
                )
                p_p7_stencil = get_p7_float32_stencil(
                    pressure.data, density.shape, 0, i + i_offset, j + j_offset, k + k_offset
                )
                cell_bx_p7_stencil = get_p7_float32_stencil(
                    cell_magnetic_field.data, density.shape, 0, i + i_offset, j + j_offset, k + k_offset
                )
                cell_by_p7_stencil = get_p7_float32_stencil(
                    cell_magnetic_field.data, density.shape, 1, i + i_offset, j + j_offset, k + k_offset
                )
                cell_bz_p7_stencil = get_p7_float32_stencil(
                    cell_magnetic_field.data, density.shape, 2, i + i_offset, j + j_offset, k + k_offset
                )
                id_p7_stencil = get_p7_uint8_stencil(
                    id_field.data, density.shape, 0, i + i_offset, j + j_offset, k + k_offset
                )

                # Apply boundary conditions
                (
                    rho_p7_stencil,
                    vx_p7_stencil,
                    vy_p7_stencil,
                    vz_p7_stencil,
                    p_p7_stencil,
                    cell_bx_p7_stencil,
                    cell_by_p7_stencil,
                    cell_bz_p7_stencil,
                ) = apply_boundary_conditions(
                    rho_p7_stencil,
                    vx_p7_stencil,
                    vy_p7_stencil,
                    vz_p7_stencil,
                    p_p7_stencil,
                    cell_bx_p7_stencil,
                    cell_by_p7_stencil,
                    cell_bz_p7_stencil,
                    id_p7_stencil,
                )

                # Get derivatives
                rho_dxyz = IdealMHDUpdate._p7_stencil_to_derivative(
                    rho_p7_stencil, density.spacing
                )
                vx_dxyz = IdealMHDUpdate._p7_stencil_to_derivative(
                    vx_p7_stencil, velocity.spacing
                )
                vy_dxyz = IdealMHDUpdate._p7_stencil_to_derivative(
                    vy_p7_stencil, velocity.spacing
                )
                vz_dxyz = IdealMHDUpdate._p7_stencil_to_derivative(
                    vz_p7_stencil, velocity.spacing
                )
                p_dxyz = IdealMHDUpdate._p7_stencil_to_derivative(
                    p_p7_stencil, pressure.spacing
                )
                cell_bx_dxyz = IdealMHDUpdate._p7_stencil_to_derivative(
                    cell_bx_p7_stencil, cell_magnetic_field.spacing
                )
                cell_by_dxyz = IdealMHDUpdate._p7_stencil_to_derivative(
                    cell_by_p7_stencil, cell_magnetic_field.spacing
                )
                cell_bz_dxyz = IdealMHDUpdate._p7_stencil_to_derivative(
                    cell_bz_p7_stencil, cell_magnetic_field.spacing
                )

                # Set stencil_
                rho_p4_stencil[c] = rho_p7_stencil[0]
                vx_p4_stencil[c] = vx_p7_stencil[0]
                vy_p4_stencil[c] = vy_p7_stencil[0]
                vz_p4_stencil[c] = vz_p7_stencil[0]
                p_p4_stencil[c] = p_p7_stencil[0]
                cell_bx_p4_stencil[c] = cell_bx_p7_stencil[0]
                cell_by_p4_stencil[c] = cell_by_p7_stencil[0]
                cell_bz_p4_stencil[c] = cell_bz_p7_stencil[0]
                id_p4_stencil[c] = id_p7_stencil[0]
                rho_p4_stencil_dxyz[c, 0] = rho_dxyz[0]
                rho_p4_stencil_dxyz[c, 1] = rho_dxyz[1]
                rho_p4_stencil_dxyz[c, 2] = rho_dxyz[2]
                vx_p4_stencil_dxyz[c, 0] = vx_dxyz[0]
                vx_p4_stencil_dxyz[c, 1] = vx_dxyz[1]
                vx_p4_stencil_dxyz[c, 2] = vx_dxyz[2]
                vy_p4_stencil_dxyz[c, 0] = vy_dxyz[0]
                vy_p4_stencil_dxyz[c, 1] = vy_dxyz[1]
                vy_p4_stencil_dxyz[c, 2] = vy_dxyz[2]
                vz_p4_stencil_dxyz[c, 0] = vz_dxyz[0]
                vz_p4_stencil_dxyz[c, 1] = vz_dxyz[1]
                vz_p4_stencil_dxyz[c, 2] = vz_dxyz[2]
                p_p4_stencil_dxyz[c, 0] = p_dxyz[0]
                p_p4_stencil_dxyz[c, 1] = p_dxyz[1]
                p_p4_stencil_dxyz[c, 2] = p_dxyz[2]
                cell_bx_p4_stencil_dxyz[c, 0] = cell_bx_dxyz[0]
                cell_bx_p4_stencil_dxyz[c, 1] = cell_bx_dxyz[1]
                cell_bx_p4_stencil_dxyz[c, 2] = cell_bx_dxyz[2]
                cell_by_pj_stencil_dxyz[c, 0] = cell_by_dxyz[0]
                cell_by_p4_stencil_dxyz[c, 1] = cell_by_dxyz[1]
                cell_by_p4_stencil_dxyz[c, 2] = cell_by_dxyz[2]
                cell_bz_p4_stencil_dxyz[c, 0] = cell_bz_dxyz[0]
                cell_bz_p4_stencil_dxyz[c, 1] = cell_bz_dxyz[1]
                cell_bz_p4_stencil_dxyz[c, 2] = cell_bz_dxyz[2]

            # Apply boundary conditions
            (
                rho_p7_stencil,
                vx_p7_stencil,
                vy_p7_stencil,
                vz_p7_stencil,
                p_p7_stencil,
                cell_bx_p7_stencil,
                cell_by_p7_stencil,
                cell_bz_p7_stencil,
            ) = apply_boundary_conditions(
                rho_p7_stencil,
                vx_p7_stencil,
                vy_p7_stencil,
                vz_p7_stencil,
                p_p7_stencil,
                cell_bx_p7_stencil,
                cell_by_p7_stencil,
                cell_bz_p7_stencil,
                id_p7_stencil,
            )

            # Apply derivative boundary conditions
            (
                rho_p7_stencil_dxyz,
                vx_p7_stencil_dxyz,
                vy_p7_stencil_dxyz,
                vz_p7_stencil_dxyz,
                p_p7_stencil_dxyz,
                cell_bx_p7_stencil_dxyz,
                cell_by_p7_stencil_dxyz,
                cell_bz_p7_stencil_dxyz,
            ) = apply_boundary_conditions_dxyz(
                rho_p7_stencil_dxyz,
                vx_p7_stencil_dxyz,
                vy_p7_stencil_dxyz,
                vz_p7_stencil_dxyz,
                p_p7_stencil_dxyz,
                cell_bx_p7_stencil_dxyz,
                cell_by_p7_stencil_dxyz,
                cell_bz_p7_stencil_dxyz,
                id_p7_stencil,
            )

            # Loop over stencil to update variables
            for c in range(7):

               # Extrapolate half time step
               (
                   rho, vx, vy, vz, p, cell_bx, cell_by, cell_bz
               ) = IdealMHDUpdate._extrapolate_half_time_step(
                    rho_p7_stencil[c],
                    rho_p7_stencil_dxyz[c],
                    vx_p7_stencil[c],
                    vx_p7_stencil_dxyz[c],
                    vy_p7_stencil[c],
                    vy_p7_stencil_dxyz[c],
                    vz_p7_stencil[c],
                    vz_p7_stencil_dxyz[c],
                    p_p7_stencil[c],
                    p_p7_stencil_dxyz[c],
                    cell_bx_p7_stencil[c],
                    cell_bx_p7_stencil_dxyz[c],
                    cell_by_p7_stencil[c],
                    cell_by_p7_stencil_dxyz[c],
                    cell_bz_p7_stencil[c],
                    cell_bz_p7_stencil_dxyz[c],
                    gamma,
                    dt,
               )

               # Set variables
               rho_p7_stencil[c] = rho
               vx_p7_stencil[c] = vx
               vy_p7_stencil[c] = vy
               vz_p7_stencil[c] = vz
               p_p7_stencil[c] = p
               cell_bx_p7_stencil[c] = cell_bx
               cell_by_p7_stencil[c] = cell_by
               cell_bz_p7_stencil[c] = cell_bz

            # Extrapolate to faces
            rho_faces = IdealMHDUpdate._stencil_to_faces(rho_p7_stencil, rho_p7_stencil_dxyz, density.spacing)
            vx_faces = IdealMHDUpdate._stencil_to_faces(vx_p7_stencil, vx_p7_stencil_dxyz, velocity.spacing)
            vy_faces = IdealMHDUpdate._stencil_to_faces(vy_p7_stencil, vy_p7_stencil_dxyz, velocity.spacing)
            vz_faces = IdealMHDUpdate._stencil_to_faces(vz_p7_stencil, vz_p7_stencil_dxyz, velocity.spacing)
            p_faces = IdealMHDUpdate._stencil_to_faces(p_p7_stencil, p_p7_stencil_dxyz, pressure.spacing)
            cell_bx_faces = IdealMHDUpdate._stencil_to_faces(cell_bx_p7_stencil, cell_bx_p7_stencil_dxyz, cell_magnetic_field.spacing)
            cell_by_faces = IdealMHDUpdate._stencil_to_faces(cell_by_p7_stencil, cell_by_p7_stencil_dxyz, cell_magnetic_field.spacing)
            cell_bz_faces = IdealMHDUpdate._stencil_to_faces(cell_bz_p7_stencil, cell_bz_p7_stencil_dxyz, cell_magnetic_field.spacing)

            # Allocate fluxes
            flux_faces_mass = _float32_faces_type()
            flux_faces_mom_x = _float32_faces_type()
            flux_faces_mom_y = _float32_faces_type()
            flux_faces_mom_z = _float32_faces_type()
            flux_faces_energy = _float32_faces_type()
            flux_faces_bx = _float32_faces_type()
            flux_faces_by = _float32_faces_type()
            flux_faces_bz = _float32_faces_type()

            # Loop over all faces
            for face in range(6):
                # Get sign and dimension
                if face == 0:
                    sign = 1.0
                    dim = 0
                    area = density.spacing[1] * density.spacing[2]
                elif face == 1:
                    sign = -1.0
                    dim = 0
                    area = density.spacing[1] * density.spacing[2]
                elif face == 2:
                    sign = 1.0
                    dim = 1
                    area = density.spacing[0] * density.spacing[2]
                elif face == 3:
                    sign = -1.0
                    dim = 1
                    area = density.spacing[0] * density.spacing[2]
                elif face == 4:
                    sign = 1.0
                    dim = 2
                    area = density.spacing[0] * density.spacing[1]
                elif face == 5:
                    sign = -1.0
                    dim = 2
                    area = density.spacing[0] * density.spacing[1]

                # Compute fluxes
                (
                    flux_mass,
                    flux_mom_x,
                    flux_mom_y,
                    flux_mom_z,
                    flux_energy,
                    flux_bx,
                    flux_by,
                    flux_bz,
                ) = IdealMHDUpdate._compute_fluxes(
                    rho_faces[2 * face + 0],
                    rho_faces[2 * face + 1],
                    vx_faces[2 * face + 0],
                    vx_faces[2 * face + 1],
                    vy_faces[2 * face + 0],
                    vy_faces[2 * face + 1],
                    vz_faces[2 * face + 0],
                    vz_faces[2 * face + 1],
                    p_faces[2 * face + 0],
                    p_faces[2 * face + 1],
                    cell_bx_faces[2 * face + 0],
                    cell_bx_faces[2 * face + 1],
                    cell_by_faces[2 * face + 0],
                    cell_by_faces[2 * face + 1],
                    cell_bz_faces[2 * face + 0],
                    cell_bz_faces[2 * face + 1],
                    dim,
                    sign,
                    area,
                    gamma,
                    dt,
                )

                # Update fluxes
                flux_faces_mass[face]   = -dt * area * flux_mass
                flux_faces_mom_x[face]  = -dt * area * flux_mom_x
                flux_faces_mom_y[face]  = -dt * area * flux_mom_y
                flux_faces_mom_z[face]  = -dt * area * flux_mom_z
                flux_faces_energy[face] = -dt * area * flux_energy
                flux_faces_bx[face] = flux_bx
                flux_faces_by[face] = flux_by
                flux_faces_bz[face] = flux_bz

            # Get total fluxes
            total_mass_flux = (
                flux_faces_mass[0]
                + flux_faces_mass[1]
                + flux_faces_mass[2]
                + flux_faces_mass[3]
                + flux_faces_mass[4]
                + flux_faces_mass[5]
            )
            total_mom_x_flux = (
                flux_faces_mom_x[0]
                + flux_faces_mom_x[1]
                + flux_faces_mom_x[2]
                + flux_faces_mom_x[3]
                + flux_faces_mom_x[4]
                + flux_faces_mom_x[5]
            )
            total_mom_y_flux = (
                flux_faces_mom_y[0]
                + flux_faces_mom_y[1]
                + flux_faces_mom_y[2]
                + flux_faces_mom_y[3]
                + flux_faces_mom_y[4]
                + flux_faces_mom_y[5]
            )
            total_mom_z_flux = (
                flux_faces_mom_z[0]
                + flux_faces_mom_z[1]
                + flux_faces_mom_z[2]
                + flux_faces_mom_z[3]
                + flux_faces_mom_z[4]
                + flux_faces_mom_z[5]
            )
            total_energy_flux = (
                flux_faces_energy[0]
                + flux_faces_energy[1]
                + flux_faces_energy[2]
                + flux_faces_energy[3]
                + flux_faces_energy[4]
                + flux_faces_energy[5]
            )

            # Update variables
            mass.data[0, i, j, k] = total_mass_flux + mass.data[0, i, j, k]
            momentum.data[0, i, j, k] = total_mom_x_flux + momentum.data[0, i, j, k]
            momentum.data[1, i, j, k] = total_mom_y_flux + momentum.data[1, i, j, k]
            momentum.data[2, i, j, k] = total_mom_z_flux + momentum.data[2, i, j, k]
            energy.data[0, i, j, k] = total_energy_flux + energy.data[0, i, j, k]

            # Store magnetic fluxes
            periodic_setting(
                flux_magnetic_field.data, flux_faces_by[0], flux_magnetic_field.shape, 0, i-1, j, k,
            )
            periodic_setting(
                flux_magnetic_field.data, flux_faces_bz[0], flux_magnetic_field.shape, 1, i-1, j, k,
            )
            periodic_setting(
                flux_magnetic_field.data, flux_faces_bx[2], flux_magnetic_field.shape, 2, i, j-1, k,
            )
            periodic_setting(
                flux_magnetic_field.data, flux_faces_bz[2], flux_magnetic_field.shape, 3, i, j-1, k,
            )
            periodic_setting(
                flux_magnetic_field.data, flux_faces_bx[4], flux_magnetic_field.shape, 4, i, j, k-1,
            )
            periodic_setting(
                flux_magnetic_field.data, flux_faces_by[4], flux_magnetic_field.shape, 5, i, j, k-1,
            )

        self._ideal_mhd_update = _ideal_mhd_update

    def __call__(
        self,
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
        cell_magnetic_field: Fieldfloat32,
        mass: Fieldfloat32,
        momentum: Fieldfloat32,
        energy: Fieldfloat32,
        flux_magnetic_field: Fieldfloat32,
        id_field: Fielduint8,
        gamma: float,
        dt: float,
    ):
        # Launch kernel
        wp.launch(
            self._ideal_mhd_update,
            inputs=[
                density,
                velocity,
                pressure,
                cell_magnetic_field,
                mass,
                momentum,
                energy,
                flux_magnetic_field,
                id_field,
                gamma,
                dt,
            ],
            dim=density.shape,
        )

        return mass, momentum, energy, flux_magnetic_field


class ConstrainedTransport(object):

    @wp.func
    def _get_curl(
        vx_1_1_1: wp.float32,
        vx_1_0_1: wp.float32,
        vx_1_1_0: wp.float32,
        vy_1_1_1: wp.float32,
        vy_0_1_1: wp.float32,
        vy_1_1_0: wp.float32,
        vz_1_1_1: wp.float32,
        vz_0_1_1: wp.float32,
        vz_1_0_1: wp.float32,
        spacing: wp.vec3f,
    ):

        # Compute curl
        curl_x = (vz_1_1_1 - vz_1_0_1) / spacing[1] - (vy_1_1_1 - vy_1_1_0) / spacing[2]
        curl_y = (vx_1_1_1 - vx_1_1_0) / spacing[2] - (vz_1_1_1 - vz_0_1_1) / spacing[0]
        curl_z = (vy_1_1_1 - vy_0_1_1) / spacing[0] - (vx_1_1_1 - vx_1_0_1) / spacing[1]

        return curl_x, curl_y, curl_z

    @wp.func
    def _compute_electric(
        flux_by_face_x_0_0_0: wp.float32,
        flux_by_face_x_0_1_0: wp.float32,
        flux_bz_face_x_0_0_0: wp.float32,
        flux_bz_face_x_0_0_1: wp.float32,
        flux_bx_face_y_0_0_0: wp.float32,
        flux_bx_face_y_1_0_0: wp.float32,
        flux_bz_face_y_0_0_0: wp.float32,
        flux_bz_face_y_0_0_1: wp.float32,
        flux_by_face_z_0_0_0: wp.float32,
        flux_by_face_z_0_1_0: wp.float32,
        flux_bx_face_z_0_0_0: wp.float32,
        flux_bx_face_z_1_0_0: wp.float32,
    ):

        # Compute electric field
        ex = 0.25 * (
            flux_bz_face_y_0_0_0 + flux_bz_face_y_0_0_1 - flux_by_face_z_0_0_0 - flux_by_face_z_0_1_0
        )
        ey = 0.25 * (
            flux_bx_face_z_0_0_0 + flux_bx_face_z_1_0_0 - flux_bz_face_x_0_0_0 - flux_bz_face_x_0_0_1
        )
        ez = 0.25 * (
            flux_by_face_x_0_0_0 + flux_by_face_x_0_1_0 - flux_bx_face_y_0_0_0 - flux_bx_face_y_1_0_0
        )

        return ex, ey, ez


    @wp.kernel
    def _constrained_transport(
        face_magnetic_field: Fieldfloat32,
        flux_magnetic_field: Fieldfloat32,
        dt: wp.float32,
    ):

        # Get index
        i, j, k = wp.tid()

        # Get stencil
        # flux by face x
        flux_by_face_x_1_0_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 0, i + 0, j - 1, k + 0)
        flux_by_face_x_1_1_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 0, i + 0, j + 0, k + 0)
        flux_by_face_x_1_2_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 0, i + 0, j + 1, k + 0)
        flux_by_face_x_0_1_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 0, i - 1, j + 0, k + 0)
        flux_by_face_x_0_2_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 0, i - 1, j + 1, k + 0)

        # flux bz face x
        flux_bz_face_x_1_1_0 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 1, i + 0, j + 0, k - 1)
        flux_bz_face_x_1_1_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 1, i + 0, j + 0, k + 0)
        flux_bz_face_x_1_1_2 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 1, i + 0, j + 0, k + 1)
        flux_bz_face_x_0_1_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 1, i - 1, j + 0, k + 0)
        flux_bz_face_x_0_1_2 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 1, i - 1, j + 0, k + 1)

        # flux bx face y
        flux_bx_face_y_0_1_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 2, i - 1, j + 0, k + 0)
        flux_bx_face_y_1_1_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 2, i + 0, j + 0, k + 0)
        flux_bx_face_y_2_1_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 2, i + 1, j + 0, k + 0)
        flux_bx_face_y_1_0_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 2, i + 0, j - 1, k + 0)
        flux_bx_face_y_2_0_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 2, i + 1, j - 1, k + 0)

        # flux bz face y
        flux_bz_face_y_1_1_0 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 3, i + 0, j + 0, k - 1)
        flux_bz_face_y_1_1_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 3, i + 0, j + 0, k + 0)
        flux_bz_face_y_1_1_2 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 3, i + 0, j + 0, k + 1)
        flux_bz_face_y_1_0_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 3, i + 0, j - 1, k + 0)
        flux_bz_face_y_1_0_2 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 3, i + 0, j - 1, k + 1)

        # flux bx face z
        flux_bx_face_z_0_1_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 4, i - 1, j + 0, k + 0)
        flux_bx_face_z_1_1_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 4, i + 0, j + 0, k + 0)
        flux_bx_face_z_2_1_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 4, i + 1, j + 0, k + 0)
        flux_bx_face_z_1_1_0 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 4, i + 0, j + 0, k - 1)
        flux_bx_face_z_2_1_0 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 4, i + 1, j + 0, k - 1)

        # flux by face z
        flux_by_face_z_1_0_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 5, i + 0, j - 1, k + 0)
        flux_by_face_z_1_1_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 5, i + 0, j + 0, k + 0)
        flux_by_face_z_1_2_1 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 5, i + 0, j + 1, k + 0)
        flux_by_face_z_1_1_0 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 5, i + 0, j + 0, k - 1)
        flux_by_face_z_1_2_0 = periodic_indexing(flux_magnetic_field.data, flux_magnetic_field.shape, 5, i + 0, j + 1, k - 1)

        # Get electric field
        ex_1_1_1 = 0.25 * (
            flux_bz_face_y_1_1_1 + flux_bz_face_y_1_1_2 - flux_by_face_z_1_1_1 - flux_by_face_z_1_2_1
        )
        ex_1_0_1 = 0.25 * (
            flux_bz_face_y_1_0_1 + flux_bz_face_y_1_0_2 - flux_by_face_z_1_0_1 - flux_by_face_z_1_1_1
        )
        ex_1_1_0 = 0.25 * (
            flux_bz_face_y_1_1_0 + flux_bz_face_y_1_1_1 - flux_by_face_z_1_1_0 - flux_by_face_z_1_2_0
        )
        ey_1_1_1 = 0.25 * (
            flux_bx_face_z_1_1_1 + flux_bx_face_z_2_1_1 - flux_bz_face_x_1_1_1 - flux_bz_face_x_1_1_2
        )
        ey_0_1_1 = 0.25 * (
            flux_bx_face_z_0_1_1 + flux_bx_face_z_1_1_1 - flux_bz_face_x_0_1_1 - flux_bz_face_x_0_1_2
        )
        ey_1_1_0 = 0.25 * (
            flux_bx_face_z_1_1_0 + flux_bx_face_z_2_1_0 - flux_bz_face_x_1_1_0 - flux_bz_face_x_1_1_1
        )
        ez_1_1_1 = 0.25 * (
            flux_by_face_x_1_1_1 + flux_by_face_x_1_2_1 - flux_bx_face_y_1_1_1 - flux_bx_face_y_2_1_1
        )
        ez_0_1_1 = 0.25 * (
            flux_by_face_x_0_1_1 + flux_by_face_x_0_2_1 - flux_bx_face_y_0_1_1 - flux_bx_face_y_1_1_1
        )
        ez_1_0_1 = 0.25 * (
            flux_by_face_x_1_0_1 + flux_by_face_x_1_1_1 - flux_bx_face_y_1_0_1 - flux_bx_face_y_2_0_1
        )

        # Compute db components
        dbx, dby, dbz = ConstrainedTransport._get_curl(
            ex_1_1_1,
            ex_1_0_1,
            ex_1_1_0,
            ey_1_1_1,
            ey_0_1_1,
            ey_1_1_0,
            ez_1_1_1,
            ez_0_1_1,
            ez_1_0_1,
            face_magnetic_field.spacing,
        )

        # Update magnetic field
        face_magnetic_field.data[0, i, j, k] = face_magnetic_field.data[0, i, j, k] + dt * dbx
        face_magnetic_field.data[1, i, j, k] = face_magnetic_field.data[1, i, j, k] + dt * dby
        face_magnetic_field.data[2, i, j, k] = face_magnetic_field.data[2, i, j, k] + dt * dbz


    def __call__(
        self,
        face_magnetic_field: Fieldfloat32,
        flux_magnetic_field: Fieldfloat32,
        dt: float,
    ):
        # Launch kernel
        wp.launch(
            self._constrained_transport,
            inputs=[
                face_magnetic_field,
                flux_magnetic_field,
                dt,
            ],
            dim=face_magnetic_field.shape,
        )

        return face_magnetic_field

class FaceMagneticFieldToCellMagneticField(Operator):

    @wp.kernel
    def _face_magnetic_field_to_cell_magnetic_field(
        face_magnetic_field: Fieldfloat32,
        cell_magnetic_field: Fieldfloat32,
    ):

        # Get index
        i, j, k = wp.tid()

        # Compute cell magnetic field
        face_b_x_1_1_1 = periodic_indexing(face_magnetic_field.data, face_magnetic_field.shape, 0, i + 0, j + 0, k + 0)
        face_b_x_0_1_1 = periodic_indexing(face_magnetic_field.data, face_magnetic_field.shape, 0, i - 1, j + 0, k + 0)
        face_b_y_1_1_1 = periodic_indexing(face_magnetic_field.data, face_magnetic_field.shape, 1, i + 0, j + 0, k + 0)
        face_b_y_1_0_1 = periodic_indexing(face_magnetic_field.data, face_magnetic_field.shape, 1, i + 0, j - 1, k + 0)
        face_b_z_1_1_1 = periodic_indexing(face_magnetic_field.data, face_magnetic_field.shape, 2, i + 0, j + 0, k + 0)
        face_b_z_1_1_0 = periodic_indexing(face_magnetic_field.data, face_magnetic_field.shape, 2, i + 0, j + 0, k - 1)

        # Set cell magnetic field
        cell_magnetic_field.data[0, i, j, k] = 0.5 * (face_b_x_1_1_1 + face_b_x_0_1_1)
        cell_magnetic_field.data[1, i, j, k] = 0.5 * (face_b_y_1_1_1 + face_b_y_1_0_1)
        cell_magnetic_field.data[2, i, j, k] = 0.5 * (face_b_z_1_1_1 + face_b_z_1_1_0)

    def __call__(
        self,
        face_magnetic_field: Fieldfloat32,
        cell_magnetic_field: Fieldfloat32,
    ):
        # Launch kernel
        wp.launch(
            self._face_magnetic_field_to_cell_magnetic_field,
            inputs=[face_magnetic_field, cell_magnetic_field],
            dim=face_magnetic_field.shape,
        )

        return cell_magnetic_field
