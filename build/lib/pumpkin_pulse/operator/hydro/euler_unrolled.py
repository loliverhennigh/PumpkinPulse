# Base class for FV euler solver

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
    get_p4_p7_float32_stencil,
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
        rho = m / volume
        vel = mom / rho / volume

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
        vx_prime = vx - dt_half * (v_dx_dot_v + p_dxyz[0] / rho)
        vy_prime = vy - dt_half * (v_dy_dot_v + p_dxyz[1] / rho)
        vz_prime = vz - dt_half * (v_dz_dot_v + p_dxyz[2] / rho)
        p_prime = p - dt_half * (gamma * p * v_dxyz_trace + v_dot_p_dxyz)

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
        avg_p = (gamma - 1.0) * (
            avg_e
            - 0.5 * (avg_mom_x**2.0 + avg_mom_y**2.0 + avg_mom_z**2.0) / avg_rho
        )

        # Compute fluxes
        flux_mass = avg_mom_x
        flux_mom_x = avg_mom_x**2.0 / avg_rho + avg_p
        flux_mom_y = avg_mom_x * avg_mom_y / avg_rho
        flux_mom_z = avg_mom_x * avg_mom_z / avg_rho
        flux_energy = avg_mom_x * (avg_e + avg_p) / avg_rho

        # Compute wave speeds
        c_l = wp.sqrt(gamma * p_l / rho_l) + wp.abs(vx_l)
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
    def _apply_fluxes(
        field: Fieldfloat32,
        flux_face_x: wp.float32,
        flux_face_y: wp.float32,
        flux_face_z: wp.float32,
        spacing: wp.vec3f,
        dt: wp.float32,
        c: wp.int32,
        i: wp.int32,
        j: wp.int32,
        k: wp.int32,
    ):

        # Subtract right cell
        periodic_atomic_add(
            field.data, -(dt * spacing[1] * spacing[2] * flux_face_x), field.shape, c, i, j, k
        )
        periodic_atomic_add(
            field.data, -(dt * spacing[0] * spacing[2] * flux_face_y), field.shape, c, i, j, k
        )
        periodic_atomic_add(
            field.data, -(dt * spacing[0] * spacing[1] * flux_face_z), field.shape, c, i, j, k
        )

        # Add left cell
        periodic_atomic_add(
            field.data, dt * spacing[1] * spacing[2] * flux_face_x, field.shape, c, i - 1, j, k
        )
        periodic_atomic_add(
            field.data, dt * spacing[0] * spacing[2] * flux_face_y, field.shape, c, i, j - 1, k
        )
        periodic_atomic_add(
            field.data, dt * spacing[0] * spacing[1] * flux_face_z, field.shape, c, i, j, k - 1
        )

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
    def _do_nothing_boundary_conditions_p4(
        rho_stencil: p4_float32_stencil_type,
        vx_stencil: p4_float32_stencil_type,
        vy_stencil: p4_float32_stencil_type,
        vz_stencil: p4_float32_stencil_type,
        p_stencil: p4_float32_stencil_type,
        id_stencil: p4_uint8_stencil_type,
    ):
        return rho_stencil, vx_stencil, vy_stencil, vz_stencil, p_stencil

    @wp.func
    def _do_nothing_boundary_conditions_p4_dxyz(
        rho_stencil_dxyz: p4_vec3f_stencil_type,
        vx_stencil_dxyz: p4_vec3f_stencil_type,
        vy_stencil_dxyz: p4_vec3f_stencil_type,
        vz_stencil_dxyz: p4_vec3f_stencil_type,
        p_stencil_dxyz: p4_vec3f_stencil_type,
        id_stencil: p4_uint8_stencil_type,
    ):
        return (
            rho_stencil_dxyz,
            vx_stencil_dxyz,
            vy_stencil_dxyz,
            vz_stencil_dxyz,
            p_stencil_dxyz,
        )

    def __init__(
        self,
        apply_boundary_conditions_p7: callable = None,
        apply_boundary_conditions_p4: callable = None,
        apply_boundary_conditions_p4_dxyz: callable = None,
        slope_limiter: callable = None,
    ):
        # Set boundary conditions functions
        if apply_boundary_conditions_p7 is None:
            apply_boundary_conditions_p7 = self._do_nothing_boundary_conditions_p7
        if apply_boundary_conditions_p4 is None:
            apply_boundary_conditions_p4 = self._do_nothing_boundary_conditions_p4
        if apply_boundary_conditions_p4_dxyz is None:
            apply_boundary_conditions_p4_dxyz = (
                self._do_nothing_boundary_conditions_p4_dxyz
            )

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

        # Generate update function
        @wp.kernel
        def _euler_update(
            density: Fieldfloat32,
            velocity: Fieldfloat32,
            pressure: Fieldfloat32,
            mass: Fieldfloat32,
            momentum: Fieldfloat32,
            energy: Fieldfloat32,
            id_field: Fielduint8,
            gamma: wp.float32,
            dt: wp.float32,
        ):
            # Get index
            i, j, k = wp.tid()

            # Check if inside domain
            if id_field.data[0, i, j, k] != wp.uint8(0):
                return

            # Get p7 stencil
            (
                rho_p7_stencil_1_1_1,
                rho_p7_stencil_0_1_1,
                rho_p7_stencil_1_0_1,
                rho_p7_stencil_1_1_0,
            ) = get_p4_p7_float32_stencil(
                density.data, density.shape, 0, i, j, k
            )
            (
                vx_p7_stencil_1_1_1,
                vx_p7_stencil_0_1_1,
                vx_p7_stencil_1_0_1,
                vx_p7_stencil_1_1_0,
            ) = get_p4_p7_float32_stencil(
                velocity.data, density.shape, 0, i, j, k
            )
            (
                vy_p7_stencil_1_1_1,
                vy_p7_stencil_0_1_1,
                vy_p7_stencil_1_0_1,
                vy_p7_stencil_1_1_0,
            ) = get_p4_p7_float32_stencil(
                velocity.data, density.shape, 1, i, j, k
            )
            (
                vz_p7_stencil_1_1_1,
                vz_p7_stencil_0_1_1,
                vz_p7_stencil_1_0_1,
                vz_p7_stencil_1_1_0,
            ) = get_p4_p7_float32_stencil(
                velocity.data, density.shape, 2, i, j, k
            )
            (
                p_p7_stencil_1_1_1,
                p_p7_stencil_0_1_1,
                p_p7_stencil_1_0_1,
                p_p7_stencil_1_1_0,
            ) = get_p4_p7_float32_stencil(
                pressure.data, density.shape, 0, i, j, k
            )

            # Get derivatives
            # center
            rho_dxyz_1_1_1 = p7_stencil_to_dxyz(rho_p7_stencil_1_1_1, density.spacing)
            vx_dxyz_1_1_1 = p7_stencil_to_dxyz(vx_p7_stencil_1_1_1, velocity.spacing)
            vy_dxyz_1_1_1 = p7_stencil_to_dxyz(vy_p7_stencil_1_1_1, velocity.spacing)
            vz_dxyz_1_1_1 = p7_stencil_to_dxyz(vz_p7_stencil_1_1_1, velocity.spacing)
            p_dxyz_1_1_1 = p7_stencil_to_dxyz(p_p7_stencil_1_1_1, pressure.spacing)

            # lower x
            rho_dxyz_0_1_1 = p7_stencil_to_dxyz(rho_p7_stencil_0_1_1, density.spacing)
            vx_dxyz_0_1_1 = p7_stencil_to_dxyz(vx_p7_stencil_0_1_1, velocity.spacing)
            vy_dxyz_0_1_1 = p7_stencil_to_dxyz(vy_p7_stencil_0_1_1, velocity.spacing)
            vz_dxyz_0_1_1 = p7_stencil_to_dxyz(vz_p7_stencil_0_1_1, velocity.spacing)
            p_dxyz_0_1_1 = p7_stencil_to_dxyz(p_p7_stencil_0_1_1, pressure.spacing)

            # lower y
            rho_dxyz_1_0_1 = p7_stencil_to_dxyz(rho_p7_stencil_1_0_1, density.spacing)
            vx_dxyz_1_0_1 = p7_stencil_to_dxyz(vx_p7_stencil_1_0_1, velocity.spacing)
            vy_dxyz_1_0_1 = p7_stencil_to_dxyz(vy_p7_stencil_1_0_1, velocity.spacing)
            vz_dxyz_1_0_1 = p7_stencil_to_dxyz(vz_p7_stencil_1_0_1, velocity.spacing)
            p_dxyz_1_0_1 = p7_stencil_to_dxyz(p_p7_stencil_1_0_1, pressure.spacing)

            # lower z
            rho_dxyz_1_1_0 = p7_stencil_to_dxyz(rho_p7_stencil_1_1_0, density.spacing)
            vx_dxyz_1_1_0 = p7_stencil_to_dxyz(vx_p7_stencil_1_1_0, velocity.spacing)
            vy_dxyz_1_1_0 = p7_stencil_to_dxyz(vy_p7_stencil_1_1_0, velocity.spacing)
            vz_dxyz_1_1_0 = p7_stencil_to_dxyz(vz_p7_stencil_1_1_0, velocity.spacing)
            p_dxyz_1_1_0 = p7_stencil_to_dxyz(p_p7_stencil_1_1_0, pressure.spacing)

            # Set center values
            rho_1_1_1 = rho_p7_stencil_1_1_1[0]
            vx_1_1_1 = vx_p7_stencil_1_1_1[0]
            vy_1_1_1 = vy_p7_stencil_1_1_1[0]
            vz_1_1_1 = vz_p7_stencil_1_1_1[0]
            p_1_1_1 = p_p7_stencil_1_1_1[0]

            # Set lower x values
            rho_0_1_1 = rho_p7_stencil_0_1_1[0]
            vx_0_1_1 = vx_p7_stencil_0_1_1[0]
            vy_0_1_1 = vy_p7_stencil_0_1_1[0]
            vz_0_1_1 = vz_p7_stencil_0_1_1[0]
            p_0_1_1 = p_p7_stencil_0_1_1[0]

            # Set lower y values
            rho_1_0_1 = rho_p7_stencil_1_0_1[0]
            vx_1_0_1 = vx_p7_stencil_1_0_1[0]
            vy_1_0_1 = vy_p7_stencil_1_0_1[0]
            vz_1_0_1 = vz_p7_stencil_1_0_1[0]
            p_1_0_1 = p_p7_stencil_1_0_1[0]

            # Set lower z values
            rho_1_1_0 = rho_p7_stencil_1_1_0[0]
            vx_1_1_0 = vx_p7_stencil_1_1_0[0]
            vy_1_1_0 = vy_p7_stencil_1_1_0[0]
            vz_1_1_0 = vz_p7_stencil_1_1_0[0]
            p_1_1_0 = p_p7_stencil_1_1_0[0]

            # Extrapolate half time step
            # center
            rho_1_1_1, vx_1_1_1, vy_1_1_1, vz_1_1_1, p_1_1_1 = EulerUpdate._extrapolate_half_time_step(
                rho_1_1_1,
                rho_dxyz_1_1_1,
                vx_1_1_1,
                vx_dxyz_1_1_1,
                vy_1_1_1,
                vy_dxyz_1_1_1,
                vz_1_1_1,
                vz_dxyz_1_1_1,
                p_1_1_1,
                p_dxyz_1_1_1,
                gamma,
                dt,
            )

            # lower x
            rho_0_1_1, vx_0_1_1, vy_0_1_1, vz_0_1_1, p_0_1_1 = EulerUpdate._extrapolate_half_time_step(
                rho_0_1_1,
                rho_dxyz_0_1_1,
                vx_0_1_1,
                vx_dxyz_0_1_1,
                vy_0_1_1,
                vy_dxyz_0_1_1,
                vz_0_1_1,
                vz_dxyz_0_1_1,
                p_0_1_1,
                p_dxyz_0_1_1,
                gamma,
                dt,
            )

            # lower y
            rho_1_0_1, vx_1_0_1, vy_1_0_1, vz_1_0_1, p_1_0_1 = EulerUpdate._extrapolate_half_time_step(
                rho_1_0_1,
                rho_dxyz_1_0_1,
                vx_1_0_1,
                vx_dxyz_1_0_1,
                vy_1_0_1,
                vy_dxyz_1_0_1,
                vz_1_0_1,
                vz_dxyz_1_0_1,
                p_1_0_1,
                p_dxyz_1_0_1,
                gamma,
                dt,
            )

            # lower z
            rho_1_1_0, vx_1_1_0, vy_1_1_0, vz_1_1_0, p_1_1_0 = EulerUpdate._extrapolate_half_time_step(
                rho_1_1_0,
                rho_dxyz_1_1_0,
                vx_1_1_0,
                vx_dxyz_1_1_0,
                vy_1_1_0,
                vy_dxyz_1_1_0,
                vz_1_1_0,
                vz_dxyz_1_1_0,
                p_1_1_0,
                p_dxyz_1_1_0,
                gamma,
                dt,
            )

            # Compute faces
            # x face
            rho_r_face_x = rho_1_1_1 - 0.5 * rho_dxyz_1_1_1[0] * density.spacing[0]
            rho_l_face_x = rho_0_1_1 + 0.5 * rho_dxyz_0_1_1[0] * density.spacing[0]
            vx_r_face_x = vx_1_1_1 - 0.5 * vx_dxyz_1_1_1[0] * velocity.spacing[0]
            vx_l_face_x = vx_0_1_1 + 0.5 * vx_dxyz_0_1_1[0] * velocity.spacing[0]
            vy_r_face_x = vy_1_1_1 - 0.5 * vy_dxyz_1_1_1[0] * velocity.spacing[0]
            vy_l_face_x = vy_0_1_1 + 0.5 * vy_dxyz_0_1_1[0] * velocity.spacing[0]
            vz_r_face_x = vz_1_1_1 - 0.5 * vz_dxyz_1_1_1[0] * velocity.spacing[0]
            vz_l_face_x = vz_0_1_1 + 0.5 * vz_dxyz_0_1_1[0] * velocity.spacing[0]
            p_r_face_x = p_1_1_1 - 0.5 * p_dxyz_1_1_1[0] * pressure.spacing[0]
            p_l_face_x = p_0_1_1 + 0.5 * p_dxyz_0_1_1[0] * pressure.spacing[0]

            # y face
            rho_r_face_y = rho_1_1_1 - 0.5 * rho_dxyz_1_1_1[1] * density.spacing[1]
            rho_l_face_y = rho_1_0_1 + 0.5 * rho_dxyz_1_0_1[1] * density.spacing[1]
            vx_r_face_y = vx_1_1_1 - 0.5 * vx_dxyz_1_1_1[1] * velocity.spacing[1]
            vx_l_face_y = vx_1_0_1 + 0.5 * vx_dxyz_1_0_1[1] * velocity.spacing[1]
            vy_r_face_y = vy_1_1_1 - 0.5 * vy_dxyz_1_1_1[1] * velocity.spacing[1]
            vy_l_face_y = vy_1_0_1 + 0.5 * vy_dxyz_1_0_1[1] * velocity.spacing[1]
            vz_r_face_y = vz_1_1_1 - 0.5 * vz_dxyz_1_1_1[1] * velocity.spacing[1]
            vz_l_face_y = vz_1_0_1 + 0.5 * vz_dxyz_1_0_1[1] * velocity.spacing[1]
            p_r_face_y = p_1_1_1 - 0.5 * p_dxyz_1_1_1[1] * pressure.spacing[1]
            p_l_face_y = p_1_0_1 + 0.5 * p_dxyz_1_0_1[1] * pressure.spacing[1]

            # z face
            rho_r_face_z = rho_1_1_1 - 0.5 * rho_dxyz_1_1_1[2] * density.spacing[2]
            rho_l_face_z = rho_1_1_0 + 0.5 * rho_dxyz_1_1_0[2] * density.spacing[2]
            vx_r_face_z = vx_1_1_1 - 0.5 * vx_dxyz_1_1_1[2] * velocity.spacing[2]
            vx_l_face_z = vx_1_1_0 + 0.5 * vx_dxyz_1_1_0[2] * velocity.spacing[2]
            vy_r_face_z = vy_1_1_1 - 0.5 * vy_dxyz_1_1_1[2] * velocity.spacing[2]
            vy_l_face_z = vy_1_1_0 + 0.5 * vy_dxyz_1_1_0[2] * velocity.spacing[2]
            vz_r_face_z = vz_1_1_1 - 0.5 * vz_dxyz_1_1_1[2] * velocity.spacing[2]
            vz_l_face_z = vz_1_1_0 + 0.5 * vz_dxyz_1_1_0[2] * velocity.spacing[2]
            p_r_face_z = p_1_1_1 - 0.5 * p_dxyz_1_1_1[2] * pressure.spacing[2]
            p_l_face_z = p_1_1_0 + 0.5 * p_dxyz_1_1_0[2] * pressure.spacing[2]

            # Compute fluxes
            (
                flux_mass_face_x,
                flux_mom_x_face_x,
                flux_mom_y_face_x,
                flux_mom_z_face_x,
                flux_energy_face_x,
            ) = EulerUpdate._compute_fluxes(
                rho_l_face_x,
                rho_r_face_x,
                vx_l_face_x,
                vx_r_face_x,
                vy_l_face_x,
                vy_r_face_x,
                vz_l_face_x,
                vz_r_face_x,
                p_l_face_x,
                p_r_face_x,
                gamma,
            )
            (
                flux_mass_face_y,
                flux_mom_y_face_y,
                flux_mom_x_face_y,
                flux_mom_z_face_y,
                flux_energy_face_y,
            ) = EulerUpdate._compute_fluxes(
                rho_l_face_y,
                rho_r_face_y,
                vy_l_face_y,
                vy_r_face_y,
                vx_l_face_y,
                vx_r_face_y,
                vz_l_face_y,
                vz_r_face_y,
                p_l_face_y,
                p_r_face_y,
                gamma,
            )
            (
                flux_mass_face_z,
                flux_mom_z_face_z,
                flux_mom_x_face_z,
                flux_mom_y_face_z,
                flux_energy_face_z,
            ) = EulerUpdate._compute_fluxes(
                rho_l_face_z,
                rho_r_face_z,
                vz_l_face_z,
                vz_r_face_z,
                vx_l_face_z,
                vx_r_face_z,
                vy_l_face_z,
                vy_r_face_z,
                p_l_face_z,
                p_r_face_z,
                gamma,
            )

            # Apply fluxes
            EulerUpdate._apply_fluxes(
                mass,
                flux_mass_face_x,
                flux_mass_face_y,
                flux_mass_face_z,
                density.spacing,
                dt,
                0,
                i,
                j,
                k,
            )
            EulerUpdate._apply_fluxes(
                momentum,
                flux_mom_x_face_x,
                flux_mom_x_face_y,
                flux_mom_x_face_z,
                velocity.spacing,
                dt,
                0,
                i,
                j,
                k,
            )
            EulerUpdate._apply_fluxes(
                momentum,
                flux_mom_y_face_x,
                flux_mom_y_face_y,
                flux_mom_y_face_z,
                velocity.spacing,
                dt,
                1,
                i,
                j,
                k,
            )
            EulerUpdate._apply_fluxes(
                momentum,
                flux_mom_z_face_x,
                flux_mom_z_face_y,
                flux_mom_z_face_z,
                velocity.spacing,
                dt,
                2,
                i,
                j,
                k,
            )
            EulerUpdate._apply_fluxes(
                energy,
                flux_energy_face_x,
                flux_energy_face_y,
                flux_energy_face_z,
                density.spacing,
                dt,
                0,
                i,
                j,
                k,
            )

        self._euler_update = _euler_update

    def __call__(
        self,
        density: Fieldfloat32,
        velocity: Fieldfloat32,
        pressure: Fieldfloat32,
        mass: Fieldfloat32,
        momentum: Fieldfloat32,
        energy: Fieldfloat32,
        id_field: Fielduint8,
        gamma: float,
        dt: float,
    ):
        # Launch kernel
        wp.launch(
            self._euler_update,
            inputs=[
                density,
                velocity,
                pressure,
                mass,
                momentum,
                energy,
                id_field,
                gamma,
                dt,
            ],
            dim=density.shape,
        )

        return mass, momentum, energy
