# Base class for electromagnetism operators

from typing import Union
import warp as wp

from sciops.operator.operator import Operator
from sciops.operator.utils.indexing import periodic_indexing, periodic_indexing_uint8

# PML ordering (36 elements)
# ( 0,  1,  2): phi_e
# ( 3,  4,  5): phi_h
# ( 6,  7,  8): psi_ex
# ( 9, 10, 11): psi_ey
# (12, 13, 14): psi_ez
# (15, 16, 17): psi_hx
# (18, 19, 20): psi_hy
# (21, 22, 23): psi_hz
# (24, 25, 26): be
# (27, 28, 29): ce
# (30, 31, 32): bh
# (33, 34, 35): ch
# Ref: https://github.com/flaport/fdtd
# Awesome resource!!! Thanks @flaport!!!!

class PMLInitializer(Operator):
    """
    Initialize PML operator
    """

    @wp.kernel
    def _initialize_pml_layer(
        pml_layer: wp.array4d(dtype=wp.float32),
        direction: wp.vec3f,
        thickness: wp.int32,
        courant_number: wp.float32,
        kk: wp.float32,
        a: wp.float32,
    ):
        # get index
        i, j, k = wp.tid()

        # float index
        ijk_f = wp.vec3f(wp.float32(i), wp.float32(j), wp.float32(k))

        # Get step
        if direction[0] == 1.0 or direction[1] == 1.0 or direction[2] == 1.0:
            step_e = wp.float32(thickness) - wp.dot(ijk_f, direction) - 0.5
            step_h = wp.float32(thickness) - wp.dot(ijk_f, direction) - 1.0
        else:
            step_e = - wp.dot(ijk_f, direction) + 0.5
            step_h = - wp.dot(ijk_f, direction) + 1.0
            
            if i == thickness - 1: # TODO: Hack to match FDTD, check if it is correct
                step_h = 0.0

        # Get sigma_e and sigma_h
        sigma_e = (40.0 * step_e ** 3.0) / (wp.float32(thickness) + 1.0) ** 4.0
        sigma_h = (40.0 * step_h ** 3.0) / (wp.float32(thickness) + 1.0) ** 4.0

        # Get vector sigma_e and sigma_h
        if direction[0] != 0.0:
            vec_sigma_e = wp.vec3f(sigma_e, 0.0, 0.0)
            vec_sigma_h = wp.vec3f(sigma_h, 0.0, 0.0)
        elif direction[1] != 0.0:
            vec_sigma_e = wp.vec3f(0.0, sigma_e, 0.0)
            vec_sigma_h = wp.vec3f(0.0, sigma_h, 0.0)
        else:
            vec_sigma_e = wp.vec3f(0.0, 0.0, sigma_e)
            vec_sigma_h = wp.vec3f(0.0, 0.0, sigma_h)

        # Get E coefficients
        be = wp.vec3f(
            wp.exp(-((vec_sigma_e[0] / kk) + a) * courant_number),
            wp.exp(-((vec_sigma_e[1] / kk) + a) * courant_number),
            wp.exp(-((vec_sigma_e[2] / kk) + a) * courant_number),
        )
        ce = wp.cw_div(
            wp.cw_mul(be - wp.vec3f(1.0, 1.0, 1.0), vec_sigma_e),
            vec_sigma_e * kk + wp.vec3f(1.0, 1.0, 1.0) * a * kk ** 2.0,
        )

        # Get H coefficients
        bh = wp.vec3f(
            wp.exp(-((vec_sigma_h[0] / kk) + a) * courant_number),
            wp.exp(-((vec_sigma_h[1] / kk) + a) * courant_number),
            wp.exp(-((vec_sigma_h[2] / kk) + a) * courant_number),
        )
        ch = wp.cw_div(
            wp.cw_mul(bh - wp.vec3f(1.0, 1.0, 1.0), vec_sigma_h),
            vec_sigma_h * kk + wp.vec3f(1.0, 1.0, 1.0) * a * kk ** 2.0,
        )

        # Set PML layer
        pml_layer[24, i, j, k] = be[0]
        pml_layer[25, i, j, k] = be[1]
        pml_layer[26, i, j, k] = be[2]
        pml_layer[27, i, j, k] = ce[0]
        pml_layer[28, i, j, k] = ce[1]
        pml_layer[29, i, j, k] = ce[2]
        pml_layer[30, i, j, k] = bh[0]
        pml_layer[31, i, j, k] = bh[1]
        pml_layer[32, i, j, k] = bh[2]
        pml_layer[33, i, j, k] = ch[0]
        pml_layer[34, i, j, k] = ch[1]
        pml_layer[35, i, j, k] = ch[2]

    def __call__(
        self,
        pml_layer: wp.array4d(dtype=wp.float32),
        direction: wp.vec3f,
        thickness: int,
        courant_number: float,
        k=1.0,
        a=1.0e-8,
    ):

        # Launch kernel
        wp.launch(
            self._initialize_pml_layer,
            inputs=[
                pml_layer,
                direction,
                thickness,
                wp.float32(courant_number),
                wp.float32(k),
                wp.float32(a),
            ],
            dim=pml_layer.shape[1:],
        )
        return pml_layer

class PMLElectricFieldUpdate(Operator):
    """
    Apply PML to electric field
    """

    @wp.func
    def _get_eps(
        id_0_0_1: wp.uint8,
        id_0_1_0: wp.uint8,
        id_0_1_1: wp.uint8,
        id_1_0_0: wp.uint8,
        id_1_0_1: wp.uint8,
        id_1_1_0: wp.uint8,
        id_1_1_1: wp.uint8,
        eps_mapping: wp.array(dtype=wp.float32),
    ) -> wp.vec3f:

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

        return wp.vec3f(eps_x, eps_y, eps_z)

    @wp.kernel
    def _pml_electric_field_update(
        electric_field: wp.array4d(dtype=wp.float32),
        pml_layer: wp.array4d(dtype=wp.float32),
        id_field: wp.array4d(dtype=wp.uint8),
        spacing: wp.vec3f,
        pml_layer_offset: wp.vec3i,
        eps_mapping: wp.array(dtype=wp.float32),
        dt: wp.float32,
    ):

        # Get index
        i, j, k = wp.tid()

        # Get shape
        shape = wp.vec3i(
            electric_field.shape[1],
            electric_field.shape[2],
            electric_field.shape[3],
        )

        # Get electromagnetic field index
        i_e = i + pml_layer_offset[0]
        j_e = j + pml_layer_offset[1]
        k_e = k + pml_layer_offset[2]

        # Get material id
        id_0_0_1 = periodic_indexing_uint8(id_field, shape, 0, i_e - 1, j_e - 1, k_e)
        id_0_1_0 = periodic_indexing_uint8(id_field, shape, 0, i_e - 1, j_e, k_e - 1)
        id_0_1_1 = periodic_indexing_uint8(id_field, shape, 0, i_e - 1, j_e, k_e)
        id_1_0_0 = periodic_indexing_uint8(id_field, shape, 0, i_e, j_e - 1, k_e - 1)
        id_1_0_1 = periodic_indexing_uint8(id_field, shape, 0, i_e, j_e - 1, k_e)
        id_1_1_0 = periodic_indexing_uint8(id_field, shape, 0, i_e, j_e, k_e - 1)
        id_1_1_1 = periodic_indexing_uint8(id_field, shape, 0, i_e, j_e, k_e)

        # Get eps
        eps = PMLElectricFieldUpdate._get_eps(
            id_0_0_1,
            id_0_1_0,
            id_0_1_1,
            id_1_0_0,
            id_1_0_1,
            id_1_1_0,
            id_1_1_1,
            eps_mapping,
        )

        # Get phi e and phi h
        phi_e = wp.vec3f(
            pml_layer[0, i, j, k],
            pml_layer[1, i, j, k],
            pml_layer[2, i, j, k],
        )

        # Get c_eh
        _denom = 2.0 * eps
        c_eh = (2.0 * dt) / wp.cw_mul(spacing, _denom)

        # Get addition to electric field
        e_add = wp.cw_mul(c_eh, phi_e)

        # Adjust electromagnetic field
        electric_field[0, i_e, j_e, k_e] += e_add[0]
        electric_field[1, i_e, j_e, k_e] += e_add[1]
        electric_field[2, i_e, j_e, k_e] += e_add[2]

    def __call__(
        self,
        electric_field: wp.array4d(dtype=wp.float32),
        pml_layer: wp.array4d(dtype=wp.float32),
        id_field: wp.array4d(dtype=wp.uint8),
        spacing: wp.vec3f,
        pml_layer_offset: wp.vec3i,
        eps_mapping: Union[wp.array, wp.constant],
        dt: float,
    ):

        # Launch kernel
        wp.launch(
            self._pml_electric_field_update,
            inputs=[
                electric_field,
                pml_layer,
                id_field,
                spacing,
                pml_layer_offset,
                eps_mapping,
                wp.float32(dt),
            ],
            dim=pml_layer.shape[1:],
        )
        return electric_field


class PMLMagneticFieldUpdate(Operator):
    """
    Apply PML operator
    """

    @wp.func
    def _get_mu(
        id_0_1_1: wp.uint8,
        id_1_0_1: wp.uint8,
        id_1_1_0: wp.uint8,
        id_1_1_1: wp.uint8,
        mu_mapping: wp.array(dtype=wp.float32),
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

        return wp.vec3f(mu_x, mu_y, mu_z)

    @wp.kernel
    def _pml_magnetic_field_update(
        magnetic_field: wp.array4d(dtype=wp.float32),
        pml_layer: wp.array4d(dtype=wp.float32),
        id_field: wp.array4d(dtype=wp.uint8),
        spacing: wp.vec3f,
        pml_layer_offset: wp.vec3i,
        mu_mapping: wp.array(dtype=wp.float32),
        dt: wp.float32,
    ):

        # Get index
        i, j, k = wp.tid()

        # Get shape
        shape = wp.vec3i(
            magnetic_field.shape[1],
            magnetic_field.shape[2],
            magnetic_field.shape[3],
        )

        # Get electromagnetic field index
        i_h = i + pml_layer_offset[0]
        j_h = j + pml_layer_offset[1]
        k_h = k + pml_layer_offset[2]

        # Get material id
        id_0_1_1 = periodic_indexing_uint8(id_field, shape, 0, i_h - 1, j_h, k_h)
        id_1_0_1 = periodic_indexing_uint8(id_field, shape, 0, i_h, j_h - 1, k_h)
        id_1_1_0 = periodic_indexing_uint8(id_field, shape, 0, i_h, j_h, k_h - 1)
        id_1_1_1 = periodic_indexing_uint8(id_field, shape, 0, i_h, j_h, k_h)

        # Get mu
        mu = PMLMagneticFieldUpdate._get_mu(
            id_0_1_1,
            id_1_0_1,
            id_1_1_0,
            id_1_1_1,
            mu_mapping,
        )

        # Get phi e and phi h
        phi_h = wp.vec3f(
            pml_layer[3, i, j, k],
            pml_layer[4, i, j, k],
            pml_layer[5, i, j, k],
        )

        # Get c_he
        _denom = 2.0 * mu
        c_he = (2.0 * dt) / wp.cw_mul(spacing, _denom)
 
        # Get addition to electric field
        h_add = wp.cw_mul(c_he, phi_h)

        # Adjust electromagnetic field
        magnetic_field[0, i_h, j_h, k_h] -= h_add[0]
        magnetic_field[1, i_h, j_h, k_h] -= h_add[1]
        magnetic_field[2, i_h, j_h, k_h] -= h_add[2]

    def __call__(
        self,
        magnetic_field: wp.array4d(dtype=wp.float32),
        pml_layer: wp.array4d(dtype=wp.float32),
        id_field: wp.array4d(dtype=wp.uint8),
        spacing: wp.vec3f,
        pml_layer_offset: wp.vec3i,
        mu_mapping: Union[wp.array, wp.constant],
        dt: float,
    ):

        # Launch kernel
        wp.launch(
            self._pml_magnetic_field_update,
            inputs=[
                magnetic_field,
                pml_layer,
                id_field,
                spacing,
                pml_layer_offset,
                mu_mapping,
                wp.float32(dt),
            ],
            dim=pml_layer.shape[1:],
        )
        return magnetic_field


class PMLPhiEUpdate(Operator):
    """
    Update PML phi_e from electric field
    """

    @wp.kernel
    def _pml_phi_e_update(
        magnetic_field: wp.array4d(dtype=wp.float32),
        pml_layer: wp.array4d(dtype=wp.float32),
        pml_layer_offset: wp.vec3i,
    ):

        # Get index
        i, j, k = wp.tid()

        # Get shape
        shape = wp.vec3i(
            magnetic_field.shape[1],
            magnetic_field.shape[2],
            magnetic_field.shape[3],
        )

        # Get electromagnetic field index
        i_h = i + pml_layer_offset[0]
        j_h = j + pml_layer_offset[1]
        k_h = k + pml_layer_offset[2]

        # Get needed vectors
        psi_ex = wp.vec3f(
            pml_layer[6, i, j, k],
            pml_layer[7, i, j, k],
            pml_layer[8, i, j, k],
        )
        psi_ey = wp.vec3f(
            pml_layer[9, i, j, k],
            pml_layer[10, i, j, k],
            pml_layer[11, i, j, k],
        )
        psi_ez = wp.vec3f(
            pml_layer[12, i, j, k],
            pml_layer[13, i, j, k],
            pml_layer[14, i, j, k],
        )
        be = wp.vec3f(
            pml_layer[24, i, j, k],
            pml_layer[25, i, j, k],
            pml_layer[26, i, j, k],
        )
        ce = wp.vec3f(
            pml_layer[27, i, j, k],
            pml_layer[28, i, j, k],
            pml_layer[29, i, j, k],
        )

        # Get h stencil
        h_x_1_1_1 = periodic_indexing(magnetic_field, shape, 0, i_h, j_h, k_h)
        h_x_1_0_1 = periodic_indexing(magnetic_field, shape, 0, i_h, j_h - 1, k_h)
        h_x_1_1_0 = periodic_indexing(magnetic_field, shape, 0, i_h, j_h, k_h - 1)
        h_y_1_1_1 = periodic_indexing(magnetic_field, shape, 1, i_h, j_h, k_h)
        h_y_0_1_1 = periodic_indexing(magnetic_field, shape, 1, i_h - 1, j_h, k_h)
        h_y_1_1_0 = periodic_indexing(magnetic_field, shape, 1, i_h, j_h, k_h - 1)
        h_z_1_1_1 = periodic_indexing(magnetic_field, shape, 2, i_h, j_h, k_h)
        h_z_0_1_1 = periodic_indexing(magnetic_field, shape, 2, i_h - 1, j_h, k_h)
        h_z_1_0_1 = periodic_indexing(magnetic_field, shape, 2, i_h, j_h - 1, k_h)

        # Update psi_e with be
        psi_ex = wp.cw_mul(be, psi_ex)
        psi_ey = wp.cw_mul(be, psi_ey)
        psi_ez = wp.cw_mul(be, psi_ez)

        # Update psi_e with h
        if i != 0:
            psi_ey[0] += (h_z_1_1_1 - h_z_0_1_1) * ce[0]
            psi_ez[0] += (h_y_1_1_1 - h_y_0_1_1) * ce[0]
        if j != 0:
            psi_ex[1] += (h_z_1_1_1 - h_z_1_0_1) * ce[1]
            psi_ez[1] += (h_x_1_1_1 - h_x_1_0_1) * ce[1]
        if k != 0:
            psi_ex[2] += (h_y_1_1_1 - h_y_1_1_0) * ce[2]
            psi_ey[2] += (h_x_1_1_1 - h_x_1_1_0) * ce[2]

        # Get phi_e
        phi_e = wp.vec3f()
        phi_e[0] = psi_ex[1] - psi_ex[2]
        phi_e[1] = psi_ey[2] - psi_ey[0]
        phi_e[2] = psi_ez[0] - psi_ez[1]

        # Update all values
        pml_layer[0, i, j, k] = phi_e[0]
        pml_layer[1, i, j, k] = phi_e[1]
        pml_layer[2, i, j, k] = phi_e[2]
        pml_layer[6, i, j, k] = psi_ex[0]
        pml_layer[7, i, j, k] = psi_ex[1]
        pml_layer[8, i, j, k] = psi_ex[2]
        pml_layer[9, i, j, k] = psi_ey[0]
        pml_layer[10, i, j, k] = psi_ey[1]
        pml_layer[11, i, j, k] = psi_ey[2]
        pml_layer[12, i, j, k] = psi_ez[0]
        pml_layer[13, i, j, k] = psi_ez[1]
        pml_layer[14, i, j, k] = psi_ez[2]

    def __call__(
        self,
        magnetic_field: wp.array4d(dtype=wp.float32),
        pml_layer: wp.array4d(dtype=wp.float32),
        pml_layer_offset: wp.vec3i,
    ):

        # Launch kernel
        wp.launch(
            self._pml_phi_e_update,
            inputs=[
                magnetic_field,
                pml_layer,
                pml_layer_offset,
            ],
            dim=pml_layer.shape[1:],
        )
        return pml_layer



class PMLPhiHUpdate(Operator):
    """
    Update PML phi_h from electric field
    """

    @wp.kernel
    def _update_pml_phi_h(
        electric_field: wp.array4d(dtype=wp.float32),
        pml_layer: wp.array4d(dtype=wp.float32),
        pml_layer_offset: wp.vec3i,
    ):

        # Get index
        i, j, k = wp.tid()

        # Get shape
        shape = wp.vec3i(
            electric_field.shape[1],
            electric_field.shape[2],
            electric_field.shape[3],
        )

        # Get electromagnetic field index
        i_e = i + pml_layer_offset[0]
        j_e = j + pml_layer_offset[1]
        k_e = k + pml_layer_offset[2]

        # Get needed vectors
        psi_hx = wp.vec3f(
            pml_layer[15, i, j, k],
            pml_layer[16, i, j, k],
            pml_layer[17, i, j, k],
        )
        psi_hy = wp.vec3f(
            pml_layer[18, i, j, k],
            pml_layer[19, i, j, k],
            pml_layer[20, i, j, k],
        )
        psi_hz = wp.vec3f(
            pml_layer[21, i, j, k],
            pml_layer[22, i, j, k],
            pml_layer[23, i, j, k],
        )
        bh = wp.vec3f(
            pml_layer[30, i, j, k],
            pml_layer[31, i, j, k],
            pml_layer[32, i, j, k],
        )
        ch = wp.vec3f(
            pml_layer[33, i, j, k],
            pml_layer[34, i, j, k],
            pml_layer[35, i, j, k],
        )

        # Get h stencil
        e_x_0_0_0 = periodic_indexing(electric_field, shape, 0, i_e, j_e, k_e)
        e_x_0_1_0 = periodic_indexing(electric_field, shape, 0, i_e, j_e + 1, k_e)
        e_x_0_0_1 = periodic_indexing(electric_field, shape, 0, i_e, j_e, k_e + 1)
        e_y_0_0_0 = periodic_indexing(electric_field, shape, 1, i_e, j_e, k_e)
        e_y_1_0_0 = periodic_indexing(electric_field, shape, 1, i_e + 1, j_e, k_e)
        e_y_0_0_1 = periodic_indexing(electric_field, shape, 1, i_e, j_e, k_e + 1)
        e_z_0_0_0 = periodic_indexing(electric_field, shape, 2, i_e, j_e, k_e)
        e_z_1_0_0 = periodic_indexing(electric_field, shape, 2, i_e + 1, j_e, k_e)
        e_z_0_1_0 = periodic_indexing(electric_field, shape, 2, i_e, j_e + 1, k_e)

        # Update psi_h with bh
        psi_hx = wp.cw_mul(bh, psi_hx)
        psi_hy = wp.cw_mul(bh, psi_hy)
        psi_hz = wp.cw_mul(bh, psi_hz)

        # Update psi_h with h
        if i != pml_layer.shape[1] - 1:
            psi_hy[0] += (e_z_1_0_0 - e_z_0_0_0) * ch[0]
            psi_hz[0] += (e_y_1_0_0 - e_y_0_0_0) * ch[0]
        if j != pml_layer.shape[2] - 1:
            psi_hx[1] += (e_z_0_1_0 - e_z_0_0_0) * ch[1]
            psi_hz[1] += (e_x_0_1_0 - e_x_0_0_0) * ch[1]
        if k != pml_layer.shape[3] - 1:
            psi_hx[2] += (e_y_0_0_1 - e_y_0_0_0) * ch[2]
            psi_hy[2] += (e_x_0_0_1 - e_x_0_0_0) * ch[2]

        # Get phi_h
        phi_h = wp.vec3f()
        phi_h[0] = psi_hx[1] - psi_hx[2]
        phi_h[1] = psi_hy[2] - psi_hy[0]
        phi_h[2] = psi_hz[0] - psi_hz[1]

        # Update all values
        pml_layer[3, i, j, k] = phi_h[0]
        pml_layer[4, i, j, k] = phi_h[1]
        pml_layer[5, i, j, k] = phi_h[2]
        pml_layer[15, i, j, k] = psi_hx[0]
        pml_layer[16, i, j, k] = psi_hx[1]
        pml_layer[17, i, j, k] = psi_hx[2]
        pml_layer[18, i, j, k] = psi_hy[0]
        pml_layer[19, i, j, k] = psi_hy[1]
        pml_layer[20, i, j, k] = psi_hy[2]
        pml_layer[21, i, j, k] = psi_hz[0]
        pml_layer[22, i, j, k] = psi_hz[1]
        pml_layer[23, i, j, k] = psi_hz[2]

    def __call__(
        self,
        electric_field: wp.array4d(dtype=wp.float32),
        pml_layer: wp.array4d(dtype=wp.float32),
        pml_layer_offset: wp.vec3i,
    ):
    
        # Launch kernel
        wp.launch(
            self._update_pml_phi_h,
            inputs=[
                electric_field,
                pml_layer,
                pml_layer_offset,
            ],
            dim=pml_layer.shape[1:],
        )
        return pml_layer
