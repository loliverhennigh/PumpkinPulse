# Plane wave hitting a sphere

import numpy as np
import warp as wp
from build123d import Rectangle, extrude, Sphere, Location, Circle, Rotation

wp.init()

from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.operator import Operator
from pumpkin_pulse.operator.electromagnetism import YeeEFieldUpdate

class PlaneWaveInitialize(Operator):

    @wp.kernel
    def _initialize_plane_wave(
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
        e_amplitude: float,
        sigma_e: float,
    ):
        # get index
        i, j, k = wp.tid()

        #

        # Get coefficients
        _denom = 2.0 * eps + sigma_e * dt
        c_ee = wp.cw_div(2.0 * eps - sigma_e * dt, _denom)
        c_eh = (2.0 * dt) / wp.cw_mul(electric_field.spacing, _denom)
        c_ej = (-2.0 * dt) / _denom

        # Get magnetic field stencil
        m_x_1_1_1 = periodic_indexing(magnetic_field.data, magnetic_field.shape, 0, i, j, k)
        m_x_1_0_1 = periodic_indexing(magnetic_field.data, magnetic_field.shape, 0, i, j - 1, k)
        m_x_1_1_0 = periodic_indexing(magnetic_field.data, magnetic_field.shape, 0, i, j, k - 1)
        m_y_1_1_1 = periodic_indexing(magnetic_field.data, magnetic_field.shape, 1, i, j, k)
        m_y_0_1_1 = periodic_indexing(magnetic_field.data, magnetic_field.shape, 1, i - 1, j, k)
        m_y_1_1_0 = periodic_indexing(magnetic_field.data, magnetic_field.shape, 1, i, j, k - 1)
        m_z_1_1_1 = periodic_indexing(magnetic_field.data, magnetic_field.shape, 2, i, j, k)
        m_z_0_1_1 = periodic_indexing(magnetic_field.data, magnetic_field.shape, 2, i - 1, j, k)
        m_z_1_0_1 = periodic_indexing(magnetic_field.data, magnetic_field.shape, 2, i, j - 1, k)

        # Get curl of magnetic field
        curl_h_x = (m_z_1_1_1 - m_z_1_0_1) - (m_y_1_1_1 - m_y_1_1_0)
        curl_h_y = (m_x_1_1_1 - m_x_1_1_0) - (m_z_1_1_1 - m_z_0_1_1)
        curl_h_z = (m_y_1_1_1 - m_y_0_1_1) - (m_x_1_1_1 - m_x_1_0_1)
        curl_h = wp.vec3(curl_h_x, curl_h_y, curl_h_z)

        # compute new electric field
        e = wp.vec3f(
            electric_field.data[0, i, j, k],
            electric_field.data[1, i, j, k],
            electric_field.data[2, i, j, k],
        )
        cur = wp.vec3f(
            impressed_current.data[0, i, j, k],
            impressed_current.data[1, i, j, k],
            impressed_current.data[2, i, j, k],
        )
        new_e = wp.cw_mul(c_ee, e) + wp.cw_mul(c_eh, curl_h) + wp.cw_mul(c_ej, cur)

        # Set electric field
        electric_field.data[0, i, j, k] = new_e[0]
        electric_field.data[1, i, j, k] = new_e[1]
        electric_field.data[2, i, j, k] = new_e[2]

    def __call__(
        self,
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
        impressed_current: Fieldfloat32,
        id_field: Fielduint8,
        eps_mapping: wp.array(dtype=wp.float32),
        sigma_e_mapping: wp.array(dtype=wp.float32),
        dt: float,
    ):
        # Launch kernel
        wp.launch(
            self._update_electric_field,
            inputs=[
                electric_field,
                magnetic_field,
                impressed_current,
                id_field,
                eps_mapping,
                sigma_e_mapping,
                dt,
            ],
            dim=electric_field.shape,
        )

        return electric_field
 
if __name__ == '__main__':

    # Define simulation parameters
    origin = (0.0, 0.0, 0.0)
    size = (1.0, 1.0, 1.0)
    dx = 0.01
    spacing = (dx, dx, dx)
    shape = (int(size[0]/dx), int(size[1]/dx), int(size[2]/dx))

    # Electric parameters
    c = 3.0e8
    eps = 8.854187817e-12
    mu = 4.0 * wp.pi * 1.0e-7
    sigma_e = 0.0
    sigma_m = 0.0
    sphere_eps = 3.0 * eps
    sphere_mu = 1.0 * mu
    sphere_sigma_e = 0.0
    sphere_sigma_m = 0.0

    # Wave parameters
    start_location = 0.1 # x location of the wave source
    frequency = 1.0e9
    amplitude = 1.0
    phase = 0.0
    polarization = 'x'
    direction = 'forward'

    # Use CFL condition to determine time step
    dt = dx / (2.0 * c)

    # Make the constructor
    constructor = Constructor()

    # Make the operators
    e_field_update = YeeEFieldUpdate()

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
    impressed_current = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )

    # Make material property mappings
    eps_mapping = wp.from_numpy(np.array([eps, sphere_eps], dtype=np.float32), dtype=wp.float32)
    mu_mapping = wp.from_numpy(np.array([mu, sphere_mu], dtype=np.float32), dtype=wp.float32)
    sigma_e_mapping = wp.from_numpy(np.array([sigma_e, sphere_sigma_e], dtype=np.float32), dtype=wp.float32)
    sigma_m_mapping = wp.from_numpy(np.array([sigma_m, sphere_sigma_m], dtype=np.float32), dtype=wp.float32)

    # Call e field update operator
    electric_field = e_field_update(
        electric_field,
        magnetic_field,
        impressed_current,
        id_field,
        eps_mapping,
        sigma_e_mapping,
        dt
    )






