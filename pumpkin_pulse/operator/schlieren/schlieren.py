# Base class for ray marching synthetic schlieren imaging

from typing import Union
import warp as wp

from pumpkin_pulse.data.field import Fieldfloat32, Fielduint8
from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.functional.indexing import periodic_indexing


class SyntheticSchlierenImaging(Operator):
    """
    Synthetic schlieren imaging operator
    """

    # Make kernels
    @wp.kernel
    def _synthetic_schlieren_imaging(
        density: Fieldfloat32,
        id_field: Fielduint8,
        image: wp.array3d(dtype=wp.float32),
        image_lower_bound: wp.vec3f, # lower bound of the image
        image_upper_bound_x: wp.vec3f, # upper bound of the image on x axis
        image_upper_bound_y: wp.vec3f, # upper bound of the image on y axis
        image_normal: wp.vec3f, # normal of the image
        max_casting_distance: wp.float32,
    ):

        # get index
        i, j = wp.tid()

        # Get image dx and dy
        image_dx = (image_upper_bound_x - image_lower_bound) / wp.float32(image.shape[0])
        image_dy = (image_upper_bound_y - image_lower_bound) / wp.float32(image.shape[1])

        # Get density upper and lower bounds
        density_lower_bound = density.origin
        density_upper_bound = density.origin + density.spacing * wp.vec3f(wp.float32(density.shape[0]), wp.float32(density.shape[1]), wp.float32(density.shape[2]))

        # Get ray dx
        ray_dx = wp.min(wp.min(density.spacing[0], density.spacing[1]), density.spacing[2])

        # Get ray position and direction
        ray_position = lower_bound + wp.float32(i) * image_dx + wp.float32(j) * image_dy
        ray_direction = image_normal

        # Start ray casting
        for _ in range(max_casting_distance):

            # Check if we are inside the density field
            if ray_position[0] < density_lower_bound[0] or ray_position[0] > density_upper_bound[0] or ray_position[1] < density_lower_bound[1] or ray_position[1] > density_upper_bound[1] or ray_position[2] < density_lower_bound[2] or ray_position[2] > density_upper_bound[2]:
                ray_position += ray_direction * ray_dx

            # Else get density value
            else:
                density_value = wp.sample(density, ray_position)

                # Set image value
                image[i, j] = density_value
                break


    def __call__(
        self,
    ):

        # Determine if impressed current is present
        if impressed_current is not None:
            # Launch kernel
            wp.launch(
                self._update_electric_field_with_impressed_current,
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
        else:
            # Launch kernel
            wp.launch(
                self._update_electric_field,
                inputs=[electric_field, magnetic_field, id_field, eps_mapping, sigma_e_mapping, dt],
                dim=electric_field.shape,
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

    @wp.func
    def _do_nothing_boundary_conditions(
        id_1_1_1: wp.uint8,
        id_0_1_1: wp.uint8,
        id_1_0_1: wp.uint8,
        id_1_1_0: wp.uint8,
        e_x_0_0_0: wp.float32,
        e_x_0_1_0: wp.float32,
        e_x_0_0_1: wp.float32,
        e_y_0_0_0: wp.float32,
        e_y_1_0_0: wp.float32,
        e_y_0_0_1: wp.float32,
        e_z_0_0_0: wp.float32,
        e_z_1_0_0: wp.float32,
        e_z_0_1_0: wp.float32,
    ):
        return (
            e_x_0_0_0,
            e_x_0_1_0,
            e_x_0_0_1,
            e_y_0_0_0,
            e_y_1_0_0,
            e_y_0_0_1,
            e_z_0_0_0,
            e_z_1_0_0,
            e_z_0_1_0,
        )

    def __init__(
        self,
        apply_boundary_conditions: callable = None,
    ):

        # Set boundary conditions functions
        if apply_boundary_conditions is None:
            apply_boundary_conditions = self._do_nothing_boundary_conditions


        @wp.kernel
        def _update_magnetic_field(
            electric_field: Fieldfloat32,
            magnetic_field: Fieldfloat32,
            id_field: Fielduint8,
            mu_mapping: wp.array(dtype=wp.float32),
            sigma_m_mapping: wp.array(dtype=wp.float32),
            dt: wp.float32,
        ):
            # get index
            i, j, k = wp.tid()
    
            # Get material id
            id_1_1_1 = periodic_indexing(id_field.data, id_field.shape, 0, i, j, k)
            id_0_1_1 = periodic_indexing(id_field.data, id_field.shape, 0, i - 1, j, k)
            id_1_0_1 = periodic_indexing(id_field.data, id_field.shape, 0, i, j - 1, k)
            id_1_1_0 = periodic_indexing(id_field.data, id_field.shape, 0, i, j, k - 1)

            # Get electric field stencil
            e_x_0_0_0 = periodic_indexing(electric_field.data, electric_field.shape, 0, i, j, k)
            e_x_0_1_0 = periodic_indexing(electric_field.data, electric_field.shape, 0, i, j + 1, k)
            e_x_0_0_1 = periodic_indexing(electric_field.data, electric_field.shape, 0, i, j, k + 1)
            e_y_0_0_0 = periodic_indexing(electric_field.data, electric_field.shape, 1, i, j, k)
            e_y_1_0_0 = periodic_indexing(electric_field.data, electric_field.shape, 1, i + 1, j, k)
            e_y_0_0_1 = periodic_indexing(electric_field.data, electric_field.shape, 1, i, j, k + 1)
            e_z_0_0_0 = periodic_indexing(electric_field.data, electric_field.shape, 2, i, j, k)
            e_z_1_0_0 = periodic_indexing(electric_field.data, electric_field.shape, 2, i + 1, j, k)
            e_z_0_1_0 = periodic_indexing(electric_field.data, electric_field.shape, 2, i, j + 1, k)

            ## Apply boundary conditions
            #(
            #    e_x_0_0_0,
            #    e_x_0_1_0,
            #    e_x_0_0_1,
            #    e_y_0_0_0,
            #    e_y_1_0_0,
            #    e_y_0_0_1,
            #    e_z_0_0_0,
            #    e_z_1_0_0,
            #    e_z_0_1_0,
            #) = apply_boundary_conditions(
            #    id_1_1_1,
            #    id_0_1_1,
            #    id_1_0_1,
            #    id_1_1_0,
            #    e_x_0_0_0,
            #    e_x_0_1_0,
            #    e_x_0_0_1,
            #    e_y_0_0_0,
            #    e_y_1_0_0,
            #    e_y_0_0_1,
            #    e_z_0_0_0,
            #    e_z_1_0_0,
            #    e_z_0_1_0,
            #)

            # get properties
            mu, sigma_m = YeeMagneticFieldUpdate._sample_magnetic_property(
                id_1_1_1, id_0_1_1, id_1_0_1, id_1_1_0, mu_mapping, sigma_m_mapping, i, j, k
            )
    
            # Get coefficients
            _denom = 2.0 * mu + sigma_m * dt
            c_hh = wp.cw_div(2.0 * mu - sigma_m * dt, _denom)
            c_he = (2.0 * dt) / wp.cw_mul(magnetic_field.spacing, _denom)
 
            # Get curl of electric field
            curl_e_x = (e_y_0_0_1 - e_y_0_0_0) - (e_z_0_1_0 - e_z_0_0_0)
            curl_e_y = (e_z_1_0_0 - e_z_0_0_0) - (e_x_0_0_1 - e_x_0_0_0)
            curl_e_z = (e_x_0_1_0 - e_x_0_0_0) - (e_y_1_0_0 - e_y_0_0_0)
            curl_e = wp.vec3(curl_e_x, curl_e_y, curl_e_z)
    
            # compute new magnetic field
            h = wp.vec3f(
                magnetic_field.data[0, i, j, k],
                magnetic_field.data[1, i, j, k],
                magnetic_field.data[2, i, j, k],
            )
            new_h = wp.cw_mul(c_hh, h) + wp.cw_mul(c_he, curl_e)
    
            # Set magnetic field
            magnetic_field.data[0, i, j, k] = new_h[0]
            magnetic_field.data[1, i, j, k] = new_h[1]
            magnetic_field.data[2, i, j, k] = new_h[2]

        self._update_magnetic_field = _update_magnetic_field

    def __call__(
        self,
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
        id_field: Fielduint8,
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
                mu_mapping,
                sigma_m_mapping,
                dt,
            ],
            dim=magnetic_field.shape,
        )
        return magnetic_field
