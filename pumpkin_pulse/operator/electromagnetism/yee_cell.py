# Base class for electromagnetism operators

from typing import Union
import warp as wp

from sciops.operator.operator import Operator
from sciops.operator.utils.indexing import periodic_indexing, periodic_indexing_uint8


class YeeElectricFieldUpdate(Operator):
    """
    Yee Cell electric field update operator
    """

    @wp.func
    def _get_eps_and_sigma_e(
        id_0_0_1: wp.uint8,
        id_0_1_0: wp.uint8,
        id_0_1_1: wp.uint8,
        id_1_0_0: wp.uint8,
        id_1_0_1: wp.uint8,
        id_1_1_0: wp.uint8,
        id_1_1_1: wp.uint8,
        eps_mapping: wp.array(dtype=wp.float32),
        sigma_e_mapping: wp.array(dtype=wp.float32),
        i: wp.int32,
        j: wp.int32,
        k: wp.int32,
    ):
        # Get eps
        eps_0_0_1 = eps_mapping[wp.int32(id_0_0_1)]
        eps_0_1_0 = eps_mapping[wp.int32(id_0_1_0)]
        eps_0_1_1 = eps_mapping[wp.int32(id_0_1_1)]
        eps_1_0_0 = eps_mapping[wp.int32(id_1_0_0)]
        eps_1_0_1 = eps_mapping[wp.int32(id_1_0_1)]
        eps_1_1_0 = eps_mapping[wp.int32(id_1_1_0)]
        eps_1_1_1 = eps_mapping[wp.int32(id_1_1_1)]
        eps_x = (eps_1_1_1 + eps_1_1_0 + eps_1_0_1 + eps_1_0_0) / 4.0
        eps_y = (eps_1_1_1 + eps_1_1_0 + eps_0_1_1 + eps_0_1_0) / 4.0
        eps_z = (eps_1_1_1 + eps_1_0_1 + eps_0_1_1 + eps_0_0_1) / 4.0

        # Get sigma_e
        sigma_e_0_0_1 = sigma_e_mapping[wp.int32(id_0_0_1)]
        sigma_e_0_1_0 = sigma_e_mapping[wp.int32(id_0_1_0)]
        sigma_e_0_1_1 = sigma_e_mapping[wp.int32(id_0_1_1)]
        sigma_e_1_0_0 = sigma_e_mapping[wp.int32(id_1_0_0)]
        sigma_e_1_0_1 = sigma_e_mapping[wp.int32(id_1_0_1)]
        sigma_e_1_1_0 = sigma_e_mapping[wp.int32(id_1_1_0)]
        sigma_e_1_1_1 = sigma_e_mapping[wp.int32(id_1_1_1)]
        sigma_e_x = (sigma_e_1_1_1 + sigma_e_1_1_0 + sigma_e_1_0_1 + sigma_e_1_0_0) / 4.0
        sigma_e_y = (sigma_e_1_1_1 + sigma_e_1_1_0 + sigma_e_0_1_1 + sigma_e_0_1_0) / 4.0
        sigma_e_z = (sigma_e_1_1_1 + sigma_e_1_0_1 + sigma_e_0_1_1 + sigma_e_0_0_1) / 4.0

        return wp.vec3(eps_x, eps_y, eps_z), wp.vec3(sigma_e_x, sigma_e_y, sigma_e_z)

    @wp.kernel
    def _update_electric_field(
        electric_field: wp.array4d(dtype=wp.float32),
        magnetic_field: wp.array4d(dtype=wp.float32),
        id_field: wp.array4d(dtype=wp.uint8),
        spacing: wp.vec3,
        eps_mapping: wp.array(dtype=wp.float32),
        sigma_e_mapping: wp.array(dtype=wp.float32),
        dt: wp.float32,
    ):
        # get index
        i, j, k = wp.tid()

        # Get shape
        shape = wp.vec3i(
            electric_field.shape[1],
            electric_field.shape[2],
            electric_field.shape[3],
        )

        # Get material id
        id_0_0_1 = periodic_indexing_uint8(id_field, shape, 0, i - 1, j - 1, k)
        id_0_1_0 = periodic_indexing_uint8(id_field, shape, 0, i - 1, j, k - 1)
        id_0_1_1 = periodic_indexing_uint8(id_field, shape, 0, i - 1, j, k)
        id_1_0_0 = periodic_indexing_uint8(id_field, shape, 0, i, j - 1, k - 1)
        id_1_0_1 = periodic_indexing_uint8(id_field, shape, 0, i, j - 1, k)
        id_1_1_0 = periodic_indexing_uint8(id_field, shape, 0, i, j, k - 1)
        id_1_1_1 = periodic_indexing_uint8(id_field, shape, 0, i, j, k)

        # Get magnetic field stencil
        m_x_1_1_1 = periodic_indexing(magnetic_field, shape, 0, i, j, k)
        m_x_1_0_1 = periodic_indexing(magnetic_field, shape, 0, i, j - 1, k)
        m_x_1_1_0 = periodic_indexing(magnetic_field, shape, 0, i, j, k - 1)
        m_y_1_1_1 = periodic_indexing(magnetic_field, shape, 1, i, j, k)
        m_y_0_1_1 = periodic_indexing(magnetic_field, shape, 1, i - 1, j, k)
        m_y_1_1_0 = periodic_indexing(magnetic_field, shape, 1, i, j, k - 1)
        m_z_1_1_1 = periodic_indexing(magnetic_field, shape, 2, i, j, k)
        m_z_0_1_1 = periodic_indexing(magnetic_field, shape, 2, i - 1, j, k)
        m_z_1_0_1 = periodic_indexing(magnetic_field, shape, 2, i, j - 1, k)

        # get properties
        eps, sigma_e = YeeElectricFieldUpdate._get_eps_and_sigma_e(
            id_0_0_1, id_0_1_0, id_0_1_1, id_1_0_0, id_1_0_1, id_1_1_0, id_1_1_1, eps_mapping, sigma_e_mapping, i, j, k
        )

        # Get coefficients
        _denom = 2.0 * eps + sigma_e * dt
        c_ee = wp.cw_div(2.0 * eps - sigma_e * dt, _denom)
        c_eh = (2.0 * dt) / wp.cw_mul(spacing, _denom)
        c_ej = (-2.0 * dt) / _denom

        # Get curl of magnetic field
        curl_h_x = (m_z_1_1_1 - m_z_1_0_1) - (m_y_1_1_1 - m_y_1_1_0)
        curl_h_y = (m_x_1_1_1 - m_x_1_1_0) - (m_z_1_1_1 - m_z_0_1_1)
        curl_h_z = (m_y_1_1_1 - m_y_0_1_1) - (m_x_1_1_1 - m_x_1_0_1)
        curl_h = wp.vec3(curl_h_x, curl_h_y, curl_h_z)

        # compute new electric field
        e = wp.vec3f(
            electric_field[0, i, j, k],
            electric_field[1, i, j, k],
            electric_field[2, i, j, k],
        )
        new_e = wp.cw_mul(c_ee, e) + wp.cw_mul(c_eh, curl_h)

        # Set electric field
        electric_field[0, i, j, k] = new_e[0]
        electric_field[1, i, j, k] = new_e[1]
        electric_field[2, i, j, k] = new_e[2]

    def __call__(
        self,
        electric_field: wp.array4d(dtype=wp.float32),
        magnetic_field: wp.array4d(dtype=wp.float32),
        id_field: wp.array4d(dtype=wp.uint8),
        spacing: wp.vec3,
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
                id_field,
                spacing,
                eps_mapping,
                sigma_e_mapping,
                dt
            ],
            dim=electric_field.shape[1:],
        )

        return electric_field


class YeeMagneticFieldUpdate(Operator):
    """
    Magnetic field update operator
    """

    @wp.func
    def _sample_magnetic_property(
        id_1_1_1: wp.uint8,
        id_0_1_1: wp.uint8,
        id_1_0_1: wp.uint8,
        id_1_1_0: wp.uint8,
        mu_mapping: wp.array(dtype=wp.float32),
        sigma_m_mapping: wp.array(dtype=wp.float32),
        i: wp.int32,
        j: wp.int32,
        k: wp.int32,
    ):

        # Get mu
        mu_1_1_1 = mu_mapping[wp.int32(id_1_1_1)]
        mu_0_1_1 = mu_mapping[wp.int32(id_0_1_1)]
        mu_1_0_1 = mu_mapping[wp.int32(id_1_0_1)]
        mu_1_1_0 = mu_mapping[wp.int32(id_1_1_0)]
        if mu_1_1_1 + mu_0_1_1 == 0.0:
            mu_x = 0.0
        else:
            mu_x = (2.0 * mu_1_1_1 * mu_0_1_1) / (mu_1_1_1 + mu_0_1_1)
        if mu_1_1_1 + mu_1_0_1 == 0.0:
            mu_y = 0.0
        else:
            mu_y = (2.0 * mu_1_1_1 * mu_1_0_1) / (mu_1_1_1 + mu_1_0_1)
        if mu_1_1_1 + mu_1_1_0 == 0.0:
            mu_z = 0.0
        else:
            mu_z = (2.0 * mu_1_1_1 * mu_1_1_0) / (mu_1_1_1 + mu_1_1_0)

        # Get sigma_m
        sigma_m_1_1_1 = sigma_m_mapping[wp.int32(id_1_1_1)]
        sigma_m_0_1_1 = sigma_m_mapping[wp.int32(id_0_1_1)]
        sigma_m_1_0_1 = sigma_m_mapping[wp.int32(id_1_0_1)]
        sigma_m_1_1_0 = sigma_m_mapping[wp.int32(id_1_1_0)]
        if sigma_m_1_1_1 + sigma_m_0_1_1 == 0.0:
            sigma_m_x = 0.0
        else:
            sigma_m_x = (2.0 * sigma_m_1_1_1 * sigma_m_0_1_1) / (sigma_m_1_1_1 + sigma_m_0_1_1)
        if sigma_m_1_1_1 + sigma_m_1_0_1 == 0.0:
            sigma_m_y = 0.0
        else:
            sigma_m_y = (2.0 * sigma_m_1_1_1 * sigma_m_1_0_1) / (sigma_m_1_1_1 + sigma_m_1_0_1)
        if sigma_m_1_1_1 + sigma_m_1_1_0 == 0.0:
            sigma_m_z = 0.0
        else:
            sigma_m_z = (2.0 * sigma_m_1_1_1 * sigma_m_1_1_0) / (sigma_m_1_1_1 + sigma_m_1_1_0)

        return wp.vec3(mu_x, mu_y, mu_z), wp.vec3(sigma_m_x, sigma_m_y, sigma_m_z)

    @wp.kernel
    def _update_magnetic_field(
        electric_field: wp.array4d(dtype=wp.float32),
        magnetic_field: wp.array4d(dtype=wp.float32),
        id_field: wp.array4d(dtype=wp.uint8),
        spacing: wp.vec3,
        mu_mapping: wp.array(dtype=wp.float32),
        sigma_m_mapping: wp.array(dtype=wp.float32),
        dt: wp.float32,
    ):
        # get index
        i, j, k = wp.tid()

        # Get shape
        shape = wp.vec3i(
            electric_field.shape[1],
            electric_field.shape[2],
            electric_field.shape[3],
        )

        # Get material id
        id_1_1_1 = periodic_indexing_uint8(id_field, shape, 0, i, j, k)
        id_0_1_1 = periodic_indexing_uint8(id_field, shape, 0, i - 1, j, k)
        id_1_0_1 = periodic_indexing_uint8(id_field, shape, 0, i, j - 1, k)
        id_1_1_0 = periodic_indexing_uint8(id_field, shape, 0, i, j, k - 1)

        # Get electric field stencil
        e_x_0_0_0 = periodic_indexing(electric_field, shape, 0, i, j, k)
        e_x_0_1_0 = periodic_indexing(electric_field, shape, 0, i, j + 1, k)
        e_x_0_0_1 = periodic_indexing(electric_field, shape, 0, i, j, k + 1)
        e_y_0_0_0 = periodic_indexing(electric_field, shape, 1, i, j, k)
        e_y_1_0_0 = periodic_indexing(electric_field, shape, 1, i + 1, j, k)
        e_y_0_0_1 = periodic_indexing(electric_field, shape, 1, i, j, k + 1)
        e_z_0_0_0 = periodic_indexing(electric_field, shape, 2, i, j, k)
        e_z_1_0_0 = periodic_indexing(electric_field, shape, 2, i + 1, j, k)
        e_z_0_1_0 = periodic_indexing(electric_field, shape, 2, i, j + 1, k)

        # get properties
        mu, sigma_m = YeeMagneticFieldUpdate._sample_magnetic_property(
            id_1_1_1, id_0_1_1, id_1_0_1, id_1_1_0, mu_mapping, sigma_m_mapping, i, j, k
        )

        # Get coefficients
        _denom = 2.0 * mu + sigma_m * dt
        c_hh = wp.cw_div(2.0 * mu - sigma_m * dt, _denom)
        c_he = (2.0 * dt) / wp.cw_mul(spacing, _denom)

        # Get curl of electric field
        curl_e_x = (e_y_0_0_1 - e_y_0_0_0) - (e_z_0_1_0 - e_z_0_0_0)
        curl_e_y = (e_z_1_0_0 - e_z_0_0_0) - (e_x_0_0_1 - e_x_0_0_0)
        curl_e_z = (e_x_0_1_0 - e_x_0_0_0) - (e_y_1_0_0 - e_y_0_0_0)
        curl_e = wp.vec3(curl_e_x, curl_e_y, curl_e_z)

        # compute new magnetic field
        h = wp.vec3f(
            magnetic_field[0, i, j, k],
            magnetic_field[1, i, j, k],
            magnetic_field[2, i, j, k],
        )
        new_h = wp.cw_mul(c_hh, h) + wp.cw_mul(c_he, curl_e)

        # Set magnetic field
        magnetic_field[0, i, j, k] = new_h[0]
        magnetic_field[1, i, j, k] = new_h[1]
        magnetic_field[2, i, j, k] = new_h[2]

    def __call__(
        self,
        electric_field: wp.array4d(dtype=wp.float32),
        magnetic_field: wp.array4d(dtype=wp.float32),
        id_field: wp.array4d(dtype=wp.uint8),
        spacing: wp.vec3,
        mu_mapping: wp.array(dtype=wp.float32),
        sigma_m_mapping: wp.array(dtype=wp.float32),
        dt: float,
    ):
        # Launch kernel
        wp.launch(
            self._update_magnetic_field,
            inputs=[
                electric_field,
                magnetic_field,
                id_field,
                spacing,
                mu_mapping,
                sigma_m_mapping,
                dt,
            ],
            dim=magnetic_field.shape[1:],
        )
        return magnetic_field

class SetElectricField(Operator):

    @wp.kernel
    def _set_electric_field(
        electric_field: wp.array4d(dtype=wp.float32),
        id_field: wp.array4d(dtype=wp.int16),
        id_value: wp.int16,
        value: wp.float32,
        direction: wp.int32,
    ):
        # Get index
        i, j, k = wp.tid()
        i += 1
        j += 1
        k += 1

        # Depending on the direction
        if direction == 0:

            # Get id value in stencil
            id_1 = id_field[0, i, j, k]
            id_2 = id_field[0, i, j, k - 1]
            id_3 = id_field[0, i, j - 1, k]
            id_4 = id_field[0, i, j - 1, k - 1]

        if direction == 1:

            # Get id value in stencil
            id_1 = id_field[0, i, j, k]
            id_2 = id_field[0, i - 1, j, k]
            id_3 = id_field[0, i, j, k - 1]
            id_4 = id_field[0, i - 1, j, k - 1]

        if direction == 2:

            # Get id value in stencil
            id_1 = id_field[0, i, j, k]
            id_2 = id_field[0, i, j - 1, k]
            id_3 = id_field[0, i - 1, j, k]
            id_4 = id_field[0, i - 1, j - 1, k]

        # Get quantity of id value in stencil that is equal to the id value
        count = 0
        if id_1 == id_value:
            count += 1
        if id_2 == id_value:
            count += 1
        if id_3 == id_value:
            count += 1
        if id_4 == id_value:
            count += 1

        # Check if count is greater than 0
        if count > 0:

            # Set the electric field
            electric_field[direction, i, j, k] = value * (wp.float32(count) / 4.0)

    def __call__(
        self,
        electric_field: wp.array4d(dtype=wp.float32),
        id_field: wp.array4d(dtype=wp.int16),
        id_value: wp.int16,
        value: float,
        direction: int,
    ):
        # Launch kernel
        wp.launch(
            self._set_electric_field,
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
