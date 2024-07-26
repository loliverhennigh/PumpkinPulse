# Base class for FV euler solver

from typing import Union
import warp as wp

from pumpkin_pulse.data.field import Fieldfloat32, Fielduint8
from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.functional.indexing import periodic_indexing

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
        energy.data[0, i, j, k] = (p / (gamma - 1.0) + 0.5 * rho * wp.dot(vel, vel)) * volume

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
        pressure.data[0, i, j, k] = (
            (e / volume) - 0.5 * rho * wp.dot(vel, vel)
        ) * (gamma - 1.0)


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
        min_spacing = wp.min(wp.min(density.spacing[0], density.spacing[1]), density.spacing[2])

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


_25p_float32_stencil_type = wp.vec(25, wp.float32)
_25p_uint8_stencil_type = wp.vec(25, wp.uint8)
_7p_float32_stencil_type = wp.vec(7, wp.float32)
_7p_uint8_stencil_type = wp.vec(7, wp.uint8)
_7p_vec3f_stencil_type = wp.mat((7, 3), wp.float32)
_float32_faces_type = wp.vec(12, wp.float32)
_float32_fluxes_type = wp.vec(6, wp.float32)

class EulerUpdate(Operator):
    """
    Update conservative variables using Euler's equations
    """

    @wp.func
    def _centeral_difference(
        v_0_1_1: wp.float32,
        v_2_1_1: wp.float32,
        v_1_0_1: wp.float32,
        v_1_2_1: wp.float32,
        v_1_1_0: wp.float32,
        v_1_1_2: wp.float32,
        spacing: wp.vec3f,
    ) -> wp.vec3:
        return wp.vec3f(
            (v_2_1_1 - v_0_1_1) / (2.0 * spacing[0]),
            (v_1_2_1 - v_1_0_1) / (2.0 * spacing[1]),
            (v_1_1_2 - v_1_1_0) / (2.0 * spacing[2]),
        )

    @wp.func
    def _minmod_slope_limiter(
        v_0: wp.float32,
        v_1: wp.float32,
        v_2: wp.float32,
        v_dx: wp.float32,
        spacing: wp.float32,
    ):

        # Set epsilon
        epsilon = 1e-8

        # Get minmod
        if v_dx == 0.0:
            denominator = 1e-8
        else:
            denominator = v_dx
        v_dx = (
            wp.max(
                0.0,
                wp.min(
                    1.0,
                    ((v_1 - v_0) / spacing) / denominator,
                ),
            )
            * v_dx
        )
        if v_dx == 0.0:
            denominator = 1e-8
        else:
            denominator = v_dx
        v_dx = (
            wp.max(
                0.0,
                wp.min(
                    1.0,
                    ((v_2 - v_1) / spacing) / denominator,
                ),
            )
            * v_dx
        )

        return v_dx

    @wp.func
    def _slope_limiter(
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

        # Break up derivatives
        v_dx = v_dxyz[0]
        v_dy = v_dxyz[1]
        v_dz = v_dxyz[2]

        # Get minmod
        v_dx = EulerUpdate._minmod_slope_limiter(
            v_0_1_1, v_1_1_1, v_2_1_1, v_dx, spacing[0]
        )
        v_dy = EulerUpdate._minmod_slope_limiter(
            v_1_0_1, v_1_1_1, v_1_2_1, v_dy, spacing[1]
        )
        v_dz = EulerUpdate._minmod_slope_limiter(
            v_1_1_0, v_1_1_1, v_1_1_2, v_dz, spacing[2]
        )

        return wp.vec3f(v_dx, v_dy, v_dz)

    @wp.func
    def _get_25p_float32_stencil(
        field: Fieldfloat32,
        c: wp.int32,
        i: wp.int32,
        j: wp.int32,
        k: wp.int32,
    ):
        # 25 point stencil
        stencil = _25p_float32_stencil_type(
            periodic_indexing(field.data, field.shape, c, i + 0, j - 1, k - 1), # Get Center Box
            periodic_indexing(field.data, field.shape, c, i - 1, j + 0, k - 1),
            periodic_indexing(field.data, field.shape, c, i + 0, j + 0, k - 1),
            periodic_indexing(field.data, field.shape, c, i + 1, j + 0, k - 1),
            periodic_indexing(field.data, field.shape, c, i + 0, j + 1, k - 1),
            periodic_indexing(field.data, field.shape, c, i - 1, j - 1, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 0, j - 1, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 1, j - 1, k + 0),
            periodic_indexing(field.data, field.shape, c, i - 1, j + 0, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 0, j + 0, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 1, j + 0, k + 0),
            periodic_indexing(field.data, field.shape, c, i - 1, j + 1, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 0, j + 1, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 1, j + 1, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 0, j - 1, k + 1),
            periodic_indexing(field.data, field.shape, c, i - 1, j + 0, k + 1),
            periodic_indexing(field.data, field.shape, c, i + 0, j + 0, k + 1),
            periodic_indexing(field.data, field.shape, c, i + 1, j + 0, k + 1),
            periodic_indexing(field.data, field.shape, c, i + 0, j + 1, k + 1),
            periodic_indexing(field.data, field.shape, c, i - 2, j + 0, k + 0), # Get edges
            periodic_indexing(field.data, field.shape, c, i + 2, j + 0, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 0, j - 2, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 0, j + 2, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 0, j + 0, k - 2),
            periodic_indexing(field.data, field.shape, c, i + 0, j + 0, k + 2),
        )
        return stencil

    @wp.func
    def _get_25p_uint8_stencil( # TODO: refactor this to use float32 implementation
        field: Fielduint8,
        c: wp.int32,
        i: wp.int32,
        j: wp.int32,
        k: wp.int32,
    ):
        # 25 point stencil
        stencil = _25p_uint8_stencil_type(
            periodic_indexing(field.data, field.shape, c, i + 0, j - 1, k - 1), # Get Center Box
            periodic_indexing(field.data, field.shape, c, i - 1, j + 0, k - 1),
            periodic_indexing(field.data, field.shape, c, i + 0, j + 0, k - 1),
            periodic_indexing(field.data, field.shape, c, i + 1, j + 0, k - 1),
            periodic_indexing(field.data, field.shape, c, i + 0, j + 1, k - 1),
            periodic_indexing(field.data, field.shape, c, i - 1, j - 1, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 0, j - 1, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 1, j - 1, k + 0),
            periodic_indexing(field.data, field.shape, c, i - 1, j + 0, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 0, j + 0, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 1, j + 0, k + 0),
            periodic_indexing(field.data, field.shape, c, i - 1, j + 1, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 0, j + 1, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 1, j + 1, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 0, j - 1, k + 1),
            periodic_indexing(field.data, field.shape, c, i - 1, j + 0, k + 1),
            periodic_indexing(field.data, field.shape, c, i + 0, j + 0, k + 1),
            periodic_indexing(field.data, field.shape, c, i + 1, j + 0, k + 1),
            periodic_indexing(field.data, field.shape, c, i + 0, j + 1, k + 1),
            periodic_indexing(field.data, field.shape, c, i - 2, j + 0, k + 0), # Get edges
            periodic_indexing(field.data, field.shape, c, i + 2, j + 0, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 0, j - 2, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 0, j + 2, k + 0),
            periodic_indexing(field.data, field.shape, c, i + 0, j + 0, k - 2),
            periodic_indexing(field.data, field.shape, c, i + 0, j + 0, k + 2),
        )
        return stencil

    @wp.func
    def _25p_float32_stencil_to_7p_float32_stencil(
        stencil: _25p_float32_stencil_type,
    ):
        stencil_7p = _7p_float32_stencil_type(
            stencil[9],
            stencil[8],
            stencil[10],
            stencil[6],
            stencil[12],
            stencil[2],
            stencil[16],
        )
        return stencil_7p

    @wp.func
    def _25p_uint8_stencil_to_7p_uint8_stencil(
        stencil: _25p_uint8_stencil_type,
    ):
        stencil_7p = _7p_uint8_stencil_type(
            stencil[9],
            stencil[8],
            stencil[10],
            stencil[6],
            stencil[12],
            stencil[2],
            stencil[16],
        )
        return stencil_7p

    @wp.func
    def _25p_stencil_to_7p_derivative_stencil(
        stencil: _25p_float32_stencil_type,
        spacing: wp.vec3f,
    ):

        # Open stencil
        v_2_1_1 = stencil[0]
        v_1_2_1 = stencil[1]
        v_2_2_1 = stencil[2]
        v_3_2_1 = stencil[3]
        v_2_3_1 = stencil[4]
        v_1_1_2 = stencil[5]
        v_2_1_2 = stencil[6]
        v_3_1_2 = stencil[7]
        v_1_2_2 = stencil[8]
        v_2_2_2 = stencil[9]
        v_3_2_2 = stencil[10]
        v_1_3_2 = stencil[11]
        v_2_3_2 = stencil[12]
        v_3_3_2 = stencil[13]
        v_2_1_3 = stencil[14]
        v_1_2_3 = stencil[15]
        v_2_2_3 = stencil[16]
        v_3_2_3 = stencil[17]
        v_2_3_3 = stencil[18]
        v_0_2_2 = stencil[19]
        v_4_2_2 = stencil[20]
        v_2_0_2 = stencil[21]
        v_2_4_2 = stencil[22]
        v_2_2_0 = stencil[23]
        v_2_2_4 = stencil[24]

        # Compute derivatives
        v_2_2_2_xyz = EulerUpdate._centeral_difference(
            v_1_2_2, v_3_2_2, v_2_1_2, v_2_3_2, v_2_2_1, v_2_2_3, spacing
        )
        v_1_2_2_xyz = EulerUpdate._centeral_difference(
            v_0_2_2, v_2_2_2, v_1_1_2, v_1_3_2, v_1_2_1, v_1_2_3, spacing 
        )
        v_3_2_2_xyz = EulerUpdate._centeral_difference(
            v_2_2_2, v_4_2_2, v_3_1_2, v_3_3_2, v_3_2_1, v_3_2_3, spacing
        )
        v_2_1_2_xyz = EulerUpdate._centeral_difference(
            v_1_1_2, v_3_1_2, v_2_0_2, v_2_2_2, v_2_1_1, v_2_1_3, spacing
        )
        v_2_3_2_xyz = EulerUpdate._centeral_difference(
            v_1_3_2, v_3_3_2, v_2_2_2, v_2_4_2, v_2_3_1, v_2_3_3, spacing
        )
        v_2_2_1_xyz = EulerUpdate._centeral_difference(
            v_1_2_1, v_3_2_1, v_2_1_1, v_2_3_1, v_2_2_0, v_2_2_2, spacing
        )
        v_2_2_3_xyz = EulerUpdate._centeral_difference(
            v_1_2_3, v_3_2_3, v_2_1_3, v_2_3_3, v_2_2_2, v_2_2_4, spacing
        )

        # Slope limiter
        v_2_2_2_xyz = EulerUpdate._slope_limiter(
            v_2_2_2, v_1_2_2, v_3_2_2, v_2_1_2, v_2_3_2, v_2_2_1, v_2_2_3, v_2_2_2_xyz, spacing
        )
        v_1_2_2_xyz = EulerUpdate._slope_limiter(
            v_1_2_2, v_0_2_2, v_2_2_2, v_1_1_2, v_1_3_2, v_1_2_1, v_1_2_3, v_1_2_2_xyz, spacing
        )
        v_3_2_2_xyz = EulerUpdate._slope_limiter(
            v_3_2_2, v_2_2_2, v_4_2_2, v_3_1_2, v_3_3_2, v_3_2_1, v_3_2_3, v_3_2_2_xyz, spacing
        )
        v_2_1_2_xyz = EulerUpdate._slope_limiter(
            v_2_1_2, v_1_1_2, v_3_1_2, v_2_0_2, v_2_2_2, v_2_1_1, v_2_1_3, v_2_1_2_xyz, spacing
        )
        v_2_3_2_xyz = EulerUpdate._slope_limiter(
            v_2_3_2, v_1_3_2, v_3_3_2, v_2_2_2, v_2_4_2, v_2_3_1, v_2_3_3, v_2_3_2_xyz, spacing
        )
        v_2_2_1_xyz = EulerUpdate._slope_limiter(
            v_2_2_1, v_1_2_1, v_3_2_1, v_2_1_1, v_2_3_1, v_2_2_0, v_2_2_2, v_2_2_1_xyz, spacing
        )
        v_2_2_3_xyz = EulerUpdate._slope_limiter(
            v_2_2_3, v_1_2_3, v_3_2_3, v_2_1_3, v_2_3_3, v_2_2_2, v_2_2_4, v_2_2_3_xyz, spacing
        )

        # Make stencil
        stencil_dxyz = _7p_vec3f_stencil_type(
            v_2_2_2_xyz[0], v_2_2_2_xyz[1], v_2_2_2_xyz[2],
            v_1_2_2_xyz[0], v_1_2_2_xyz[1], v_1_2_2_xyz[2],
            v_3_2_2_xyz[0], v_3_2_2_xyz[1], v_3_2_2_xyz[2],
            v_2_1_2_xyz[0], v_2_1_2_xyz[1], v_2_1_2_xyz[2],
            v_2_3_2_xyz[0], v_2_3_2_xyz[1], v_2_3_2_xyz[2],
            v_2_2_1_xyz[0], v_2_2_1_xyz[1], v_2_2_1_xyz[2],
            v_2_2_3_xyz[0], v_2_2_3_xyz[1], v_2_2_3_xyz[2],
        )

        return stencil_dxyz

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
    def _stencil_to_faces(
        v_stencil: _7p_float32_stencil_type,
        v_stencil_dxyz: _7p_vec3f_stencil_type,
        spacing: wp.vec3f,
    ):

        # Make faces
        faces = _float32_faces_type()

        # Set faces (out, in)
        # Lower x
        faces[0] = v_stencil[0] - 0.5 * spacing[0] * v_stencil_dxyz[0, 0]
        faces[1] = v_stencil[1] + 0.5 * spacing[0] * v_stencil_dxyz[1, 0]

        # Upper x
        faces[2] = v_stencil[0] + 0.5 * spacing[0] * v_stencil_dxyz[0, 0]
        faces[3] = v_stencil[2] - 0.5 * spacing[0] * v_stencil_dxyz[2, 0]

        # Lower y
        faces[4] = v_stencil[0] - 0.5 * spacing[1] * v_stencil_dxyz[0, 1]
        faces[5] = v_stencil[3] + 0.5 * spacing[1] * v_stencil_dxyz[3, 1]

        # Upper y
        faces[6] = v_stencil[0] + 0.5 * spacing[1] * v_stencil_dxyz[0, 1]
        faces[7] = v_stencil[4] - 0.5 * spacing[1] * v_stencil_dxyz[4, 1]

        # Lower z
        faces[8] = v_stencil[0] - 0.5 * spacing[2] * v_stencil_dxyz[0, 2]
        faces[9] = v_stencil[5] + 0.5 * spacing[2] * v_stencil_dxyz[5, 2]

        # Upper z
        faces[10] = v_stencil[0] + 0.5 * spacing[2] * v_stencil_dxyz[0, 2]
        faces[11] = v_stencil[6] - 0.5 * spacing[2] * v_stencil_dxyz[6, 2]

        return faces

    @wp.func
    def _compute_fluxes(
        rho_out: wp.float32,
        rho_in: wp.float32,
        vx_out: wp.float32,
        vx_in: wp.float32,
        vy_out: wp.float32,
        vy_in: wp.float32,
        vz_out: wp.float32,
        vz_in: wp.float32,
        p_out: wp.float32,
        p_in: wp.float32,
        dim: wp.int32,
        sign: wp.float32,
        area: wp.float32,
        gamma: wp.float32,
        dt: wp.float32,
    ):

        # Compute energies
        e_out = p_out / (gamma - 1.0) + 0.5 * rho_out * (vx_out ** 2.0 + vy_out ** 2.0 + vz_out ** 2.0)
        e_in = p_in / (gamma - 1.0) + 0.5 * rho_in * (vx_in ** 2.0 + vy_in ** 2.0 + vz_in ** 2.0)

        # Compute averages
        avg_rho = 0.5 * (rho_out + rho_in)
        avg_mom_x = 0.5 * (rho_out * vx_out + rho_in * vx_in)
        avg_mom_y = 0.5 * (rho_out * vy_out + rho_in * vy_in)
        avg_mom_z = 0.5 * (rho_out * vz_out + rho_in * vz_in)
        avg_e = 0.5 * (e_out + e_in)
        avg_p = (gamma - 1.0) * (avg_e - 0.5 * (avg_mom_x ** 2.0 + avg_mom_y ** 2.0 + avg_mom_z ** 2.0) / avg_rho)

        # Compute wave speeds
        if dim == 0:
            c_out = wp.sqrt(gamma * p_out / rho_out) + wp.abs(vx_out)
            c_in = wp.sqrt(gamma * p_in / rho_in) + wp.abs(vx_in)
        elif dim == 1:
            c_out = wp.sqrt(gamma * p_out / rho_out) + wp.abs(vy_out)
            c_in = wp.sqrt(gamma * p_in / rho_in) + wp.abs(vy_in)
        elif dim == 2:
            c_out = wp.sqrt(gamma * p_out / rho_out) + wp.abs(vz_out)
            c_in = wp.sqrt(gamma * p_in / rho_in) + wp.abs(vz_in)
        c = wp.max(c_out, c_in)

        # Compute fluxes
        if dim == 0:
            llf_flux_mass = avg_mom_x
            llf_flux_mom_x = avg_mom_x ** 2.0 / avg_rho + avg_p
            llf_flux_mom_y = avg_mom_x * avg_mom_y / avg_rho
            llf_flux_mom_z = avg_mom_x * avg_mom_z / avg_rho
            llf_flux_energy = avg_mom_x * (avg_e + avg_p) / avg_rho
        elif dim == 1:
            llf_flux_mass = avg_mom_y
            llf_flux_mom_x = avg_mom_y * avg_mom_x / avg_rho
            llf_flux_mom_y = avg_mom_y ** 2.0 / avg_rho + avg_p
            llf_flux_mom_z = avg_mom_y * avg_mom_z / avg_rho
            llf_flux_energy = avg_mom_y * (avg_e + avg_p) / avg_rho
        elif dim == 2:
            llf_flux_mass = avg_mom_z
            llf_flux_mom_x = avg_mom_z * avg_mom_x / avg_rho
            llf_flux_mom_y = avg_mom_z * avg_mom_y / avg_rho
            llf_flux_mom_z = avg_mom_z ** 2.0 / avg_rho + avg_p
            llf_flux_energy = avg_mom_z * (avg_e + avg_p) / avg_rho

        # Stabilizing diffusion term
        c_flux_mass = c * 0.5 * (rho_in - rho_out)
        c_flux_mom_x = c * 0.5 * (rho_in * vx_in - rho_out * vx_out)
        c_flux_mom_y = c * 0.5 * (rho_in * vy_in - rho_out * vy_out)
        c_flux_mom_z = c * 0.5 * (rho_in * vz_in - rho_out * vz_out)
        c_flux_energy = c * 0.5 * (e_in - e_out)

        # Compute fluxes
        flux_mass = dt * area * (sign * llf_flux_mass + c_flux_mass)
        flux_mom_x = dt * area * (sign * llf_flux_mom_x + c_flux_mom_x)
        flux_mom_y = dt * area * (sign * llf_flux_mom_y + c_flux_mom_y)
        flux_mom_z = dt * area * (sign * llf_flux_mom_z + c_flux_mom_z)
        flux_energy = dt * area * (sign * llf_flux_energy + c_flux_energy)

        return flux_mass, flux_mom_x, flux_mom_y, flux_mom_z, flux_energy

    @wp.func
    def _do_nothing_boundary_conditions(
        rho_stencil: _25p_float32_stencil_type,
        vx_stencil: _25p_float32_stencil_type,
        vy_stencil: _25p_float32_stencil_type,
        vz_stencil: _25p_float32_stencil_type,
        p_stencil: _25p_float32_stencil_type,
        id_stencil: _25p_uint8_stencil_type,
    ):
        return rho_stencil, vx_stencil, vy_stencil, vz_stencil, p_stencil

    @wp.func
    def _do_nothing_boundary_conditions_dxyz(
        rho_stencil_dxyz: _7p_vec3f_stencil_type,
        vx_stencil_dxyz: _7p_vec3f_stencil_type,
        vy_stencil_dxyz: _7p_vec3f_stencil_type,
        vz_stencil_dxyz: _7p_vec3f_stencil_type,
        p_stencil_dxyz: _7p_vec3f_stencil_type,
        id_stencil: _7p_uint8_stencil_type,
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
        apply_boundary_conditions: callable = None,
        apply_boundary_conditions_dxyz: callable = None,
    ):

        # Set boundary conditions functions
        if apply_boundary_conditions is None:
            apply_boundary_conditions = self._do_nothing_boundary_conditions
        if apply_boundary_conditions_dxyz is None:
            apply_boundary_conditions_dxyz = self._do_nothing_boundary_conditions_dxyz

        # Generate kernel
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

            # Get 25p stencil
            rho_25p_stencil = EulerUpdate._get_25p_float32_stencil(density, 0, i, j, k)
            vx_25p_stencil = EulerUpdate._get_25p_float32_stencil(velocity, 0, i, j, k)
            vy_25p_stencil = EulerUpdate._get_25p_float32_stencil(velocity, 1, i, j, k)
            vz_25p_stencil = EulerUpdate._get_25p_float32_stencil(velocity, 2, i, j, k)
            p_25p_stencil = EulerUpdate._get_25p_float32_stencil(pressure, 0, i, j, k)
            id_25p_stencil = EulerUpdate._get_25p_uint8_stencil(id_field, 0, i, j, k)

            # Apply boundary conditions
            rho_25p_stencil, vx_25p_stencil, vy_25p_stencil, vz_25p_stencil, p_25p_stencil = apply_boundary_conditions(
                rho_25p_stencil,
                vx_25p_stencil,
                vy_25p_stencil,
                vz_25p_stencil,
                p_25p_stencil,
                id_25p_stencil
            )

            # Get derivatives
            rho_7p_stencil_dxyz = EulerUpdate._25p_stencil_to_7p_derivative_stencil(rho_25p_stencil, density.spacing)
            vx_7p_stencil_dxyz = EulerUpdate._25p_stencil_to_7p_derivative_stencil(vx_25p_stencil, velocity.spacing)
            vy_7p_stencil_dxyz = EulerUpdate._25p_stencil_to_7p_derivative_stencil(vy_25p_stencil, velocity.spacing)
            vz_7p_stencil_dxyz = EulerUpdate._25p_stencil_to_7p_derivative_stencil(vz_25p_stencil, velocity.spacing)
            p_7p_stencil_dxyz = EulerUpdate._25p_stencil_to_7p_derivative_stencil(p_25p_stencil, pressure.spacing)

            # Get 7 point stencil from 25 point stencil
            rho_7p_stencil = EulerUpdate._25p_float32_stencil_to_7p_float32_stencil(rho_25p_stencil)
            vx_7p_stencil = EulerUpdate._25p_float32_stencil_to_7p_float32_stencil(vx_25p_stencil)
            vy_7p_stencil = EulerUpdate._25p_float32_stencil_to_7p_float32_stencil(vy_25p_stencil)
            vz_7p_stencil = EulerUpdate._25p_float32_stencil_to_7p_float32_stencil(vz_25p_stencil)
            p_7p_stencil = EulerUpdate._25p_float32_stencil_to_7p_float32_stencil(p_25p_stencil)
            id_7p_stencil = EulerUpdate._25p_uint8_stencil_to_7p_uint8_stencil(id_25p_stencil)

            # Apply derivative boundary conditions
            rho_7p_stencil_dxyz, vx_7p_stencil_dxyz, vy_7p_stencil_dxyz, vz_7p_stencil_dxyz, p_7p_stencil_dxyz = apply_boundary_conditions_dxyz(
                rho_7p_stencil_dxyz,
                vx_7p_stencil_dxyz,
                vy_7p_stencil_dxyz,
                vz_7p_stencil_dxyz,
                p_7p_stencil_dxyz,
                id_7p_stencil
            )

            # Loop over stencil to update variables
            for c in range(7):

                # Extrapolate half time step
                rho, vx, vy, vz, p = EulerUpdate._extrapolate_half_time_step(
                    rho_7p_stencil[c],
                    rho_7p_stencil_dxyz[c],
                    vx_7p_stencil[c],
                    vx_7p_stencil_dxyz[c],
                    vy_7p_stencil[c],
                    vy_7p_stencil_dxyz[c],
                    vz_7p_stencil[c],
                    vz_7p_stencil_dxyz[c],
                    p_7p_stencil[c],
                    p_7p_stencil_dxyz[c],
                    gamma,
                    dt,
                )

                # Set variables
                rho_7p_stencil[c] = rho
                vx_7p_stencil[c] = vx
                vy_7p_stencil[c] = vy
                vz_7p_stencil[c] = vz
                p_7p_stencil[c] = p

            # Get faces
            rho_faces = EulerUpdate._stencil_to_faces(rho_7p_stencil, rho_7p_stencil_dxyz, density.spacing)
            vx_faces = EulerUpdate._stencil_to_faces(vx_7p_stencil, vx_7p_stencil_dxyz, velocity.spacing)
            vy_faces = EulerUpdate._stencil_to_faces(vy_7p_stencil, vy_7p_stencil_dxyz, velocity.spacing)
            vz_faces = EulerUpdate._stencil_to_faces(vz_7p_stencil, vz_7p_stencil_dxyz, velocity.spacing)
            p_faces = EulerUpdate._stencil_to_faces(p_7p_stencil, p_7p_stencil_dxyz, pressure.spacing)

            # Allocate fluxes
            flux_faces_mass = _float32_faces_type()
            flux_faces_mom_x = _float32_faces_type()
            flux_faces_mom_y = _float32_faces_type()
            flux_faces_mom_z = _float32_faces_type()
            flux_faces_energy = _float32_faces_type()

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
                flux_mass, flux_mom_x, flux_mom_y, flux_mom_z, flux_energy = EulerUpdate._compute_fluxes(
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
                        dim,
                        sign,
                        area,
                        gamma,
                        dt,
                    )

                # Update fluxes
                flux_faces_mass[face] = flux_mass
                flux_faces_mom_x[face] = flux_mom_x
                flux_faces_mom_y[face] = flux_mom_y
                flux_faces_mom_z[face] = flux_mom_z
                flux_faces_energy[face] = flux_energy

            # Update variables
            for face in range(6):
                mass.data[0, i, j, k] = flux_faces_mass[face] + mass.data[0, i, j, k]
                momentum.data[0, i, j, k] = flux_faces_mom_x[face] + momentum.data[0, i, j, k]
                momentum.data[1, i, j, k] = flux_faces_mom_y[face] + momentum.data[1, i, j, k]
                momentum.data[2, i, j, k] = flux_faces_mom_z[face] + momentum.data[2, i, j, k]
                energy.data[0, i, j, k] = flux_faces_energy[face] + energy.data[0, i, j, k]

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
