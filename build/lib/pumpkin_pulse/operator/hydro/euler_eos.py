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


eos_type = wp.vec(5, wp.float32)
eos_dxyz_type = wp.mat((5, 3), wp.float32)
class TwoKernelEulerUpdate(Operator):
    """
    Update conservative variables using Euler's equations
    """

    @wp.func
    def _extrapolate_half_time_step(
        prim: eos_type,
        prim_dxyz: eos_dxyz_type,
        gamma: wp.float32,
        dt: wp.float32,
    ):

        # Unpack primitive variables
        rho = prim[0]
        vx = prim[1]
        vy = prim[2]
        vz = prim[3]
        p = prim[4]
        rho_dxyz = prim_dxyz[0]
        vx_dxyz = prim_dxyz[1]
        vy_dxyz = prim_dxyz[2]
        vz_dxyz = prim_dxyz[3]
        p_dxyz = prim_dxyz[4]

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

        return eos_type(rho_prime, vx_prime, vy_prime, vz_prime, p_prime)

    @wp.func
    def _compute_fluxes(
        prim_l: eos_type,
        prim_r: eos_type,
        gamma: wp.float32,
    ):

        # Unpack primitive variables
        rho_l = prim_l[0]
        vx_l = prim_l[1]
        vy_l = prim_l[2]
        vz_l = prim_l[3]
        p_l = prim_l[4]
        rho_r = prim_r[0]
        vx_r = prim_r[1]
        vy_r = prim_r[2]
        vz_r = prim_r[3]
        p_r = prim_r[4]

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

        return eos_type(flux_mass, flux_mom_x, flux_mom_y, flux_mom_z, flux_energy)

    def __init__(
        self,
        slope_limiter: callable = None,
    ):

        """
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
        """

        # Generate update function
        @wp.kernel
        def _compute_faces( 
            primitives: Fieldfloat32,
            primitives_dxyz: Fieldfloat32,
            gamma: wp.float32,
            dt: wp.float32,
        ):
            # Get index
            i, j, k = wp.tid()

            # Get center values
            center_prim = eos_type()
            for v in range(5):
                center_prim[v] = primitives.data[v, i, j, k]

            # Get derivatives
            center_prim_dxyz = eos_dxyz_type()
            for v in range(5):
                center_prim_dxyz[v, 0] = (
                    periodic_indexing(primitives.data, primitives.shape, v, i + 1, j, k)
                    - periodic_indexing(primitives.data, primitives.shape, v, i - 1, j, k)
                ) / (2.0 * primitives.spacing[0])
                center_prim_dxyz[v, 1] = (
                    periodic_indexing(primitives.data, primitives.shape, v, i, j + 1, k)
                    - periodic_indexing(primitives.data, primitives.shape, v, i, j - 1, k)
                ) / (2.0 * primitives.spacing[1])
                center_prim_dxyz[v, 2] = (
                    periodic_indexing(primitives.data, primitives.shape, v, i, j, k + 1)
                    - periodic_indexing(primitives.data, primitives.shape, v, i, j, k - 1)
                ) / (2.0 * primitives.spacing[2])


            # Extrapolate half time step
            prim = TwoKernelEulerUpdate._extrapolate_half_time_step(
                center_prim,
                center_prim_dxyz,
                gamma,
                dt,
            )

            # Set primitive variables
            for v in range(5):
                primitives.data[v, i, j, k] = prim[v]
                for d in range(3):
                    primitives_dxyz.data[3 * v + d, i, j, k] = center_prim_dxyz[v, d]

        self._compute_faces = _compute_faces

        # Generate update function
        @wp.kernel
        def _euler_update(
            primitives: Fieldfloat32,
            primitives_dxyz: Fieldfloat32,
            conservative: Fieldfloat32,
            gamma: wp.float32,
            dt: wp.float32,
        ):
            # Get index
            d, i, j, k = wp.tid()

            # Get i, j, k offsets depending on face
            i_offset = wp.max(1 - d, 0)
            j_offset = wp.max(1 - wp.abs(d - 1), 0)
            k_offset = wp.max(d - 1, 0)

            # Get center values and offsets
            prim_l = eos_type()
            prim_r = eos_type()
            for v in range(5):
                prim_l[v] = primitives.data[v, i, j, k] - 0.5 * primitives_dxyz.data[3 * v + d, i, j, k] * primitives.spacing[d]
                prim_r[v] = primitives.data[v, i + i_offset, j + j_offset, k + k_offset] + 0.5 * primitives_dxyz.data[3 * v + d, i + i_offset, j + j_offset, k + k_offset] * primitives.spacing[d]

            # Compute faces
            flux_eos = TwoKernelEulerUpdate._compute_fluxes(
                prim_l,
                prim_r,
                gamma,
            )

            # Apply fluxes
            for v in range(5):
                periodic_atomic_add(
                    conservative.data,
                    -dt * flux_eos[v],
                    conservative.shape,
                    v,
                    i,
                    j,
                    k,
                )
                periodic_atomic_add(
                    conservative.data,
                    dt * flux_eos[v],
                    conservative.shape,
                    v,
                    i + i_offset,
                    j + j_offset,
                    k + k_offset,
                )

        self._euler_update = _euler_update


    def __call__(
        self,
        primitives: Fieldfloat32,
        primitives_dxyz: Fieldfloat32,
        conservative: Union[Fieldfloat32, None],
        gamma: float,
        dt: float,
    ):
        # Launch kernel
        wp.launch(
            self._compute_faces,
            inputs=[
                primitives,
                primitives_dxyz,
                gamma,
                dt,
            ],
            dim=list(primitives.shape),
        )

        # Launch kernel
        wp.launch(
            self._euler_update,
            inputs=[
                primitives,
                primitives_dxyz,
                conservative,
                gamma,
                dt,
            ],
            dim=[3] + list(primitives.shape),
        )

        return conservative
