# Base class for FV euler solver

from typing import Union
import warp as wp

from pumpkin_pulse.data.field import Fieldfloat32, Fielduint8
from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.functional.indexing import periodic_indexing, periodic_indexing_uint8, periodic_atomic_add
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


class PrimitiveToConservative(Operator):
    """
    Convert primitive variables to conservative variables
    """

    @wp.kernel
    def _primitive_to_conservative(
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
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

        # Make sure density and pressure are positive
        rho = wp.max(rho, 0.0)
        p = wp.max(p, 0.0)

        # Set mass
        mass.data[0, i, j, k] = rho * volume

        # Set momentum
        momentum.data[0, i, j, k] = rho * vel[0] * volume
        momentum.data[1, i, j, k] = rho * vel[1] * volume
        momentum.data[2, i, j, k] = rho * vel[2] * volume

        # Set energy
        energy.data[0, i, j, k] = (
            p / (gamma - 1.0) + 0.5 * rho * wp.dot(vel, vel)
        ) * volume

    def __call__(
        self,
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
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

        # Get primitive variables
        if m <= 0.0:
            rho = 0.0
            vel = wp.vec3f(0.0, 0.0, 0.0)
        else:
            rho = m / volume
            vel = mom / rho / volume

        # If energy is negative, set it to zero
        e = wp.max(e, 0.0)

        # Set density
        density.data[0, i, j, k] = rho

        # Set velocity
        velocity.data[0, i, j, k] = vel[0]
        velocity.data[1, i, j, k] = vel[1]
        velocity.data[2, i, j, k] = vel[2]

        # Set pressure
        pressure.data[0, i, j, k] = ((e / volume) - 0.5 * rho * wp.dot(vel, vel)) * (
            gamma - 1.0
        )

    def __call__(
        self,
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
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
    Get time step based on CFL condition for Euler's equations
    """

    @wp.kernel
    def _get_time_step(
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
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

        # Get CFL
        if rho <= 0.0:
            cfl = courant_factor * min_spacing / wp.length(vel)
        else:
            cfl = courant_factor * min_spacing / (wp.sqrt(gamma * p / rho) + wp.length(vel))

        # Update time step
        wp.atomic_min(time_step, 0, cfl)

    def __call__(
        self,
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
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
                id_field,
                time_step,
                courant_factor,
                gamma,
            ],
            dim=density.shape,
        )

        return time_step.numpy()[0]


class EulerUpdate(Operator):
    """
    Update conservative variables using Euler's equations
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
        gamma: wp.float32,
        dt: wp.float32,
    ):

        # Make sure density and pressure are positive
        rho = wp.max(rho, 0.0)
        p = wp.max(p, 0.0)

        # Needed values
        dt_half = 0.5 * dt
        v_dxyz_trace = vx_dxyz[0] + vy_dxyz[1] + vz_dxyz[2]
        v_dot_rho_dxyz = vx * rho_dxyz[0] + vy * rho_dxyz[1] + vz * rho_dxyz[2]
        v_dx_dot_v = vx * vx_dxyz[0] + vy * vx_dxyz[1] + vz * vx_dxyz[2]
        v_dy_dot_v = vx * vy_dxyz[0] + vy * vy_dxyz[1] + vz * vy_dxyz[2]
        v_dz_dot_v = vx * vz_dxyz[0] + vy * vz_dxyz[1] + vz * vz_dxyz[2]
        v_dot_p_dxyz = vx * p_dxyz[0] + vy * p_dxyz[1] + vz * p_dxyz[2]

        # Interpolated values
        rho_prime = rho - dt_half * (v_dot_rho_dxyz + rho * v_dxyz_trace)
        if rho == 0.0:
            vx_prime = 0.0
            vy_prime = 0.0
            vz_prime = 0.0
        else:
            vx_prime = vx - dt_half * (v_dx_dot_v + p_dxyz[0] / rho)
            vy_prime = vy - dt_half * (v_dy_dot_v + p_dxyz[1] / rho)
            vz_prime = vz - dt_half * (v_dz_dot_v + p_dxyz[2] / rho)
        p_prime = p - dt_half * (gamma * p * v_dxyz_trace + v_dot_p_dxyz)

        # Make sure velocity is not nan
        if rho == 0.0:
            vx_prime = 0.0
            vy_prime = 0.0
            vz_prime = 0.0

        return rho_prime, vx_prime, vy_prime, vz_prime, p_prime

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
        gamma: wp.float32,
    ):

        # Make sure density and pressure are positive
        rho_l = wp.max(rho_l, 0.0)
        rho_r = wp.max(rho_r, 0.0)
        p_l = wp.max(p_l, 0.0)
        p_r = wp.max(p_r, 0.0)

        # Compute energies
        e_l = p_l / (gamma - 1.0) + 0.5 * rho_l * (
            vx_l**2.0 + vy_l**2.0 + vz_l**2.0
        )
        e_r = p_r / (gamma - 1.0) + 0.5 * rho_r * (
            vx_r**2.0 + vy_r**2.0 + vz_r**2.0
        )

        # Compute averages
        avg_rho = 0.5 * (rho_l + rho_r)
        avg_mom_x = 0.5 * (rho_l * vx_l + rho_r * vx_r)
        avg_mom_y = 0.5 * (rho_l * vy_l + rho_r * vy_r)
        avg_mom_z = 0.5 * (rho_l * vz_l + rho_r * vz_r)
        avg_e = 0.5 * (e_l + e_r)
        if avg_rho == 0.0:
            avg_p = (gamma - 1.0) * avg_e
        else:
            avg_p = (gamma - 1.0) * (
                avg_e
                - 0.5 * (avg_mom_x**2.0 + avg_mom_y**2.0 + avg_mom_z**2.0) / avg_rho
            )

        # Compute fluxes
        flux_mass = avg_mom_x
        if avg_rho == 0.0:
            flux_mom_x = avg_p
            flux_mom_y = 0.0
            flux_mom_z = 0.0
            flux_energy = 0.0
        else:
            flux_mom_x = avg_mom_x**2.0 / avg_rho + avg_p
            flux_mom_y = avg_mom_x * avg_mom_y / avg_rho
            flux_mom_z = avg_mom_x * avg_mom_z / avg_rho
            flux_energy = avg_mom_x * (avg_e + avg_p) / avg_rho

        # Compute wave speeds
        if rho_l == 0.0:
            c_l = wp.abs(vx_l)
        else:
            c_l = wp.sqrt(gamma * p_l / rho_l) + wp.abs(vx_l)
        if rho_r == 0.0:
            c_r = wp.abs(vx_r)
        else:
            c_r = wp.sqrt(gamma * p_r / rho_r) + wp.abs(vx_r)
        c = wp.max(c_l, c_r)

        # Stabilizing diffusion term
        flux_mass -= c * 0.5 * (rho_l - rho_r)
        flux_mom_x -= c * 0.5 * (rho_l * vx_l - rho_r * vx_r)
        flux_mom_y -= c * 0.5 * (rho_l * vy_l - rho_r * vy_r)
        flux_mom_z -= c * 0.5 * (rho_l * vz_l - rho_r * vz_r)
        flux_energy -= c * 0.5 * (e_l - e_r)

        return flux_mass, flux_mom_x, flux_mom_y, flux_mom_z, flux_energy

    @wp.func
    def _do_nothing_boundary_conditions_p7(
        rho_stencil: p7_float32_stencil_type,
        vx_stencil: p7_float32_stencil_type,
        vy_stencil: p7_float32_stencil_type,
        vz_stencil: p7_float32_stencil_type,
        p_stencil: p7_float32_stencil_type,
        id_stencil: p7_uint8_stencil_type,
    ):
        return rho_stencil, vx_stencil, vy_stencil, vz_stencil, p_stencil

    @wp.func
    def _do_nothing_boundary_conditions_faces(
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
        id_l: wp.uint8,
        id_r: wp.uint8,
    ):
        return (
            rho_l, rho_r, vx_l, vx_r, vy_l, vy_r, vz_l, vz_r, p_l, p_r
        )

    def __init__(
        self,
        apply_boundary_conditions_p7: callable = None,
        apply_boundary_conditions_faces: callable = None,
        slope_limiter: callable = None,
        fluid_id: wp.uint8 = wp.uint8(0),
    ):

        # Set boundary conditions functions
        if apply_boundary_conditions_p7 is None:
            apply_boundary_conditions_p7 = self._do_nothing_boundary_conditions_p7
        if apply_boundary_conditions_faces is None:
            apply_boundary_conditions_faces = self._do_nothing_boundary_conditions_faces

        # Set slope limiter
        if slope_limiter is None:
            slope_limiter = minmod_slope_limiter

        # Set fluid id
        fluid_id = wp.constant(fluid_id)

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

        # Generate update function
        @wp.kernel
        def _compute_derivatives(
            density: Fieldfloat32,
            velocity: Fieldfloat32,
            pressure: Fieldfloat32,
            density_dxyz: Fieldfloat32,
            velocity_dxyz: Fieldfloat32,
            pressure_dxyz: Fieldfloat32,
            id_field: Fielduint8,
            gamma: wp.float32,
            dt: wp.float32,
        ):
            # Get index
            i, j, k = wp.tid()

            # Get index for id field
            id_field_i = i + id_field.offset[0] - density.offset[0]
            id_field_j = j + id_field.offset[1] - density.offset[1]
            id_field_k = k + id_field.offset[2] - density.offset[2]

            # Get center values
            center_rho = density.data[0, i, j, k]
            center_vx = velocity.data[0, i, j, k]
            center_vy = velocity.data[1, i, j, k]
            center_vz = velocity.data[2, i, j, k]
            center_p = pressure.data[0, i, j, k]
            center_id = id_field.data[0, id_field_i, id_field_j, id_field_k]

            # Check if inside domain
            if center_id != fluid_id:
                return

            # Loop through directions
            rho_dxyz = wp.vec3f()
            vx_dxyz = wp.vec3f()
            vy_dxyz = wp.vec3f()
            vz_dxyz = wp.vec3f()
            p_dxyz = wp.vec3f()
            for d in range(3):

                # Get offsets
                i_offset = wp.max(1 - d, 0)
                j_offset = wp.max(1 - wp.abs(d - 1), 0)
                k_offset = wp.max(d - 1, 0)

                # Get stencils
                lower_rho = periodic_indexing(density.data, density.shape, 0, i - i_offset, j - j_offset, k - k_offset)
                upper_rho = periodic_indexing(density.data, density.shape, 0, i + i_offset, j + j_offset, k + k_offset)
                lower_vx = periodic_indexing(velocity.data, velocity.shape, 0, i - i_offset, j - j_offset, k - k_offset)
                upper_vx = periodic_indexing(velocity.data, velocity.shape, 0, i + i_offset, j + j_offset, k + k_offset)
                lower_vy = periodic_indexing(velocity.data, velocity.shape, 1, i - i_offset, j - j_offset, k - k_offset)
                upper_vy = periodic_indexing(velocity.data, velocity.shape, 1, i + i_offset, j + j_offset, k + k_offset)
                lower_vz = periodic_indexing(velocity.data, velocity.shape, 2, i - i_offset, j - j_offset, k - k_offset)
                upper_vz = periodic_indexing(velocity.data, velocity.shape, 2, i + i_offset, j + j_offset, k + k_offset)
                lower_p = periodic_indexing(pressure.data, pressure.shape, 0, i - i_offset, j - j_offset, k - k_offset)
                upper_p = periodic_indexing(pressure.data, pressure.shape, 0, i + i_offset, j + j_offset, k + k_offset)

                # Get derivatives
                rho_dxyz[d] = (upper_rho - lower_rho) / density.spacing[d]
                vx_dxyz[d] = (upper_vx - lower_vx) / velocity.spacing[d]
                vy_dxyz[d] = (upper_vy - lower_vy) / velocity.spacing[d]
                vz_dxyz[d] = (upper_vz - lower_vz) / velocity.spacing[d]
                p_dxyz[d] = (upper_p - lower_p) / pressure.spacing[d]

            ## Get derivatives
            #rho_dxyz = wp.vec3f()
            #vx_dxyz = wp.vec3f()
            #vy_dxyz = wp.vec3f()
            #vz_dxyz = wp.vec3f()
            #p_dxyz = wp.vec3f()
            #for d in range(3):
            #    rho_dxyz[d] = (rho_stencil[2 * d + 2] - rho_stencil[2 * d + 1]) / density.spacing[d]
            #    vx_dxyz[d] = (vx_stencil[2 * d + 2] - vx_stencil[2 * d + 1]) / velocity.spacing[d]
            #    vy_dxyz[d] = (vy_stencil[2 * d + 2] - vy_stencil[2 * d + 1]) / velocity.spacing[d]
            #    vz_dxyz[d] = (vz_stencil[2 * d + 2] - vz_stencil[2 * d + 1]) / velocity.spacing[d]
            #    p_dxyz[d] = (p_stencil[2 * d + 2] - p_stencil[2 * d + 1]) / pressure.spacing[d]
            #rho_dxyz = p7_stencil_to_dxyz(rho_stencil, density.spacing)
            #vx_dxyz = p7_stencil_to_dxyz(vx_stencil, velocity.spacing)
            #vy_dxyz = p7_stencil_to_dxyz(vy_stencil, velocity.spacing)
            #vz_dxyz = p7_stencil_to_dxyz(vz_stencil, velocity.spacing)
            #p_dxyz = p7_stencil_to_dxyz(p_stencil, pressure.spacing)

            # Extrapolate half time step
            rho, vx, vy, vz, p = EulerUpdate._extrapolate_half_time_step(
                center_rho,
                rho_dxyz,
                center_vx,
                vx_dxyz,
                center_vy,
                vy_dxyz,
                center_vz,
                vz_dxyz,
                center_p,
                p_dxyz,
                gamma,
                dt,
            )

            # Set primitive variables
            density.data[0, i, j, k] = rho
            velocity.data[0, i, j, k] = vx
            velocity.data[1, i, j, k] = vy
            velocity.data[2, i, j, k] = vz
            pressure.data[0, i, j, k] = p

            # Set density derivatives
            density_dxyz.data[0, i, j, k] = rho_dxyz[0]
            density_dxyz.data[1, i, j, k] = rho_dxyz[1]
            density_dxyz.data[2, i, j, k] = rho_dxyz[2]

            ## Set velocity derivatives
            #velocity_dxyz.data[0, i, j, k] = vx_dxyz[0]
            #velocity_dxyz.data[1, i, j, k] = vx_dxyz[1]
            #velocity_dxyz.data[2, i, j, k] = vx_dxyz[2]
            #velocity_dxyz.data[3, i, j, k] = vy_dxyz[0]
            #velocity_dxyz.data[4, i, j, k] = vy_dxyz[1]
            #velocity_dxyz.data[5, i, j, k] = vy_dxyz[2]
            #velocity_dxyz.data[6, i, j, k] = vz_dxyz[0]
            #velocity_dxyz.data[7, i, j, k] = vz_dxyz[1]
            #velocity_dxyz.data[8, i, j, k] = vz_dxyz[2]

            ## Set pressure derivatives
            #pressure_dxyz.data[0, i, j, k] = p_dxyz[0]
            #pressure_dxyz.data[1, i, j, k] = p_dxyz[1]
            #pressure_dxyz.data[2, i, j, k] = p_dxyz[2]

        self._compute_derivatives = _compute_derivatives

        # Generate update function
        @wp.kernel
        def _euler_update(
            density: Fieldfloat32,
            velocity: Fieldfloat32,
            pressure: Fieldfloat32,
            density_dxyz: Fieldfloat32,
            velocity_dxyz: Fieldfloat32,
            pressure_dxyz: Fieldfloat32,
            mass: Fieldfloat32,
            momentum: Fieldfloat32,
            energy: Fieldfloat32,
            id_field: Fielduint8,
            gamma: wp.float32,
            dt: wp.float32,
        ):

            # Get index
            d, i, j, k = wp.tid()

            # Get index for id field
            id_field_i = i + id_field.offset[0] - density.offset[0]
            id_field_j = j + id_field.offset[1] - density.offset[1]
            id_field_k = k + id_field.offset[2] - density.offset[2]

            # Get i, j, k offsets depending on face
            i_offset = wp.max(1 - d, 0)
            j_offset = wp.max(1 - wp.abs(d - 1), 0)
            k_offset = wp.max(d - 1, 0)

            # Get id for left and right cells
            id_l = periodic_indexing_uint8(id_field.data, id_field.shape, 0, id_field_i + i_offset, id_field_j + j_offset, id_field_k + k_offset)
            id_r = id_field.data[0, id_field_i, id_field_j, id_field_k]

            # Check if one of the cells is fluid
            if id_l != fluid_id and id_r != fluid_id:
                return

            # Get left values
            rho_l = (
                periodic_indexing(density.data, density.shape, 0, i + i_offset, j + j_offset, k + k_offset)
                - periodic_indexing(density_dxyz.data, density.shape, d, i + i_offset, j + j_offset, k + k_offset)
                * density.spacing[d]
                * 0.5
            )
            vx_l = (
                periodic_indexing(velocity.data, velocity.shape, 0, i + i_offset, j + j_offset, k + k_offset)
                - periodic_indexing(velocity_dxyz.data, velocity.shape, d, i + i_offset, j + j_offset, k + k_offset)
                * velocity.spacing[d]
                * 0.5
            )
            vy_l = (
                periodic_indexing(velocity.data, velocity.shape, 1, i + i_offset, j + j_offset, k + k_offset)
                - periodic_indexing(velocity_dxyz.data, velocity.shape, d + 3, i + i_offset, j + j_offset, k + k_offset)
                * velocity.spacing[d]
                * 0.5
            )
            vz_l = (
                periodic_indexing(velocity.data, velocity.shape, 2, i + i_offset, j + j_offset, k + k_offset)
                - periodic_indexing(velocity_dxyz.data, velocity.shape, d + 6, i + i_offset, j + j_offset, k + k_offset)
                * velocity.spacing[d]
                * 0.5
            )
            p_l = (
                periodic_indexing(pressure.data, pressure.shape, 0, i + i_offset, j + j_offset, k + k_offset)
                - periodic_indexing(pressure_dxyz.data, pressure.shape, d, i + i_offset, j + j_offset, k + k_offset)
                * pressure.spacing[d]
                * 0.5
            )

            # Get right values
            rho_r = (
                density.data[0, i, j, k]
                + density_dxyz.data[d, i, j, k] * density.spacing[d] * 0.5
            )
            vx_r = (
                velocity.data[0, i, j, k]
                + velocity_dxyz.data[d, i, j, k] * velocity.spacing[d] * 0.5
            )
            vy_r = (
                velocity.data[1, i, j, k]
                + velocity_dxyz.data[d + 3, i, j, k] * velocity.spacing[d] * 0.5
            )
            vz_r = (
                velocity.data[2, i, j, k]
                + velocity_dxyz.data[d + 6, i, j, k] * velocity.spacing[d] * 0.5
            )
            p_r = (
                pressure.data[0, i, j, k]
                + pressure_dxyz.data[d, i, j, k] * pressure.spacing[d] * 0.5
            )

            # Apply boundary conditions
            rho_l, rho_r, vx_l, vx_r, vy_l, vy_r, vz_l, vz_r, p_l, p_r = apply_boundary_conditions_faces(
                rho_l,
                rho_r,
                vx_l,
                vx_r,
                vy_l,
                vy_r,
                vz_l,
                vz_r,
                p_l,
                p_r,
                id_l,
                id_r,
            )

            # Compute faces
            flux_mass, flux_mom_x, flux_mom_y, flux_mom_z, flux_energy = EulerUpdate._compute_fluxes(
                rho_l,
                rho_r,
                vx_l,
                vx_r,
                vy_l,
                vy_r,
                vz_l,
                vz_r,
                p_l,
                p_r,
                gamma,
            )

            # Get surface area
            if d == 0:
                volume = density.spacing[1] * density.spacing[2]
            elif d == 1:
                volume = density.spacing[0] * density.spacing[2]
            else:
                volume = density.spacing[0] * density.spacing[1]

            # Apply fluxes left
            if id_l == fluid_id:
                periodic_atomic_add(
                    mass.data,
                    -dt * volume * flux_mass,
                    mass.shape,
                    0,
                    i,
                    j,
                    k,
                )
                periodic_atomic_add(
                    momentum.data,
                    -dt * volume * flux_mom_x,
                    momentum.shape,
                    0,
                    i,
                    j,
                    k,
                )
                periodic_atomic_add(
                    momentum.data,
                    -dt * volume * flux_mom_y,
                    momentum.shape,
                    1,
                    i,
                    j,
                    k,
                )
                periodic_atomic_add(
                    momentum.data,
                    -dt * volume * flux_mom_z,
                    momentum.shape,
                    2,
                    i,
                    j,
                    k,
                )
                periodic_atomic_add(
                    energy.data,
                    -dt * volume * flux_energy,
                    energy.shape,
                    0,
                    i,
                    j,
                    k,
                )

            # Apply fluxes right
            if id_r == fluid_id:
                periodic_atomic_add(
                    mass.data,
                    dt * volume * flux_mass,
                    mass.shape,
                    0,
                    i + i_offset,
                    j + j_offset,
                    k + k_offset,
                )
                periodic_atomic_add(
                    momentum.data,
                    dt * volume * flux_mom_x,
                    momentum.shape,
                    0,
                    i + i_offset,
                    j + j_offset,
                    k + k_offset,
                )
                periodic_atomic_add(
                    momentum.data,
                    dt * volume * flux_mom_y,
                    momentum.shape,
                    1,
                    i + i_offset,
                    j + j_offset,
                    k + k_offset,
                )
                periodic_atomic_add(
                    momentum.data,
                    dt * volume * flux_mom_z,
                    momentum.shape,
                    2,
                    i + i_offset,
                    j + j_offset,
                    k + k_offset,
                )
                periodic_atomic_add(
                    energy.data,
                    dt * volume * flux_energy,
                    energy.shape,
                    0,
                    i + i_offset,
                    j + j_offset,
                    k + k_offset,
                )

        self._euler_update = _euler_update


    def __call__(
        self,
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
        density_dxyz: Fieldfloat32,
        velocity_dxyz: Fieldfloat32,
        pressure_dxyz: Fieldfloat32,
        mass: Fieldfloat32,
        momentum: Fieldfloat32,
        energy: Fieldfloat32,
        id_field: Fielduint8,
        gamma: float,
        dt: float,
    ):
        # Launch kernel
        wp.launch(
            self._compute_derivatives,
            inputs=[
                density,
                velocity,
                pressure,
                density_dxyz,
                velocity_dxyz,
                pressure_dxyz,
                id_field,
                gamma,
                dt,
            ],
            dim=list(density.shape),
        )

        # Launch kernel
        wp.launch(
            self._euler_update,
            inputs=[
                density,
                velocity,
                pressure,
                density_dxyz,
                velocity_dxyz,
                pressure_dxyz,
                mass,
                momentum,
                energy,
                id_field,
                gamma,
                dt,
            ],
            dim=[3] + list(density.shape),
        )

        return mass, momentum, energy
