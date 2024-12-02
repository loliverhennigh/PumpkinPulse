# Base class for electromagnetism operators

from typing import Union
import warp as wp

from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.compute_backend import ComputeBackend
from pumpkin_pulse.material import Material


class EMSetBoundary(Operator):
    """
    Boundary condition operator

    TODO: Currently only supports Dirichlet boundary conditions and set to 0
    """

    # Make kernel for setting boundary conditions
    @wp.kernel
    def _set_ghost_cells(
        electric_field: wp.array4d(dtype=wp.float32),
        magnetic_field: wp.array4d(dtype=wp.float32),
        field_shape: wp.vec3i,
        spacing: wp.vec3f,
        dt: wp.float32,
    ):
        # get index
        index = wp.tid()

        # Get max index for each face
        x_min = field_shape[1] * field_shape[2]
        x_max = 2 * field_shape[1] * field_shape[2]
        y_min = x_max + field_shape[0] * field_shape[2]
        y_max = y_min + field_shape[0] * field_shape[2]
        z_min = y_max + field_shape[0] * field_shape[1]
        z_max = z_min + field_shape[0] * field_shape[1]

        # Go through all faces
        # x_min
        if index < x_min:
            # Get i, j, k
            i = 0
            j = wp.int32(index / field_shape[2])
            k = wp.int32(index % field_shape[2])

            # Set electric field
            electric_field[0, i, j, k] = 0.0
            electric_field[1, i, j, k] = 0.0
            electric_field[2, i, j, k] = 0.0

            # Set magnetic field
            magnetic_field[0, i, j, k] = 0.0
            magnetic_field[1, i, j, k] = 0.0
            magnetic_field[2, i, j, k] = 0.0

        # x_max
        elif index < x_max:
            # Get i, j, k
            i = field_shape[0] - 1
            j = wp.int32((index - x_min) / field_shape[2])
            j = wp.int32((index - x_min) / field_shape[2])
            k = wp.int32((index - x_min) % field_shape[2])

            # Set electric field
            electric_field[0, i, j, k] = 0.0
            electric_field[1, i, j, k] = 0.0
            electric_field[2, i, j, k] = 0.0

            # Set magnetic field
            magnetic_field[0, i, j, k] = 0.0
            magnetic_field[1, i, j, k] = 0.0
            magnetic_field[2, i, j, k] = 0.0

        # y_min
        elif index < y_min:
            # Get i, j, k
            j = 0
            i = wp.int32((index - x_max) / field_shape[2])
            k = wp.int32((index - x_max) % field_shape[2])

            # Set electric field
            electric_field[0, i, j, k] = 0.0
            electric_field[1, i, j, k] = 0.0
            electric_field[2, i, j, k] = 0.0

            # Set magnetic field
            magnetic_field[0, i, j, k] = 0.0
            magnetic_field[1, i, j, k] = 0.0
            magnetic_field[2, i, j, k] = 0.0

        # y_max
        elif index < y_max:
            # Get i, j, k
            j = field_shape[1] - 1
            i = wp.int32((index - y_min) / field_shape[2])
            k = wp.int32((index - y_min) % field_shape[2])

            # Set electric field
            electric_field[0, i, j, k] = 0.0
            electric_field[1, i, j, k] = 0.0
            electric_field[2, i, j, k] = 0.0

            # Set magnetic field
            magnetic_field[0, i, j, k] = 0.0
            magnetic_field[1, i, j, k] = 0.0
            magnetic_field[2, i, j, k] = 0.0

        # z_min
        elif index < z_min:
            # Get i, j, k
            k = 0
            i = wp.int32((index - y_max) / field_shape[1])
            j = wp.int32((index - y_max) % field_shape[1])

            # Set electric field
            electric_field[0, i, j, k] = 0.0
            electric_field[1, i, j, k] = 0.0
            electric_field[2, i, j, k] = 0.0

            # Set magnetic field
            magnetic_field[0, i, j, k] = 0.0
            magnetic_field[1, i, j, k] = 0.0
            magnetic_field[2, i, j, k] = 0.0

        # z_max
        else:
            # Get i, j, k
            k = field_shape[2] - 1
            i = wp.int32((index - z_min) / field_shape[1])
            j = wp.int32((index - z_min) % field_shape[1])

            # Set electric field
            electric_field[0, i, j, k] = 0.0
            electric_field[1, i, j, k] = 0.0
            electric_field[2, i, j, k] = 0.0

            # Set magnetic field
            magnetic_field[0, i, j, k] = 0.0
            magnetic_field[1, i, j, k] = 0.0
            magnetic_field[2, i, j, k] = 0.0

    def __call__(
        self,
        electric_field: wp.array4d(dtype=wp.float32),
        magnetic_field: wp.array4d(dtype=wp.float32),
        spacing: Union[float, tuple[float, float, float]],
        dt: float,
    ):
        # Determin number of ghost cells
        num_cells = 2 * (
            (electric_field.shape[1] * electric_field.shape[2])
            + (electric_field.shape[0] * electric_field.shape[2])
            + (electric_field.shape[0] * electric_field.shape[1])
        )

        # Launch kernel
        wp.launch(
            self._set_ghost_cells,
            inputs=[
                electric_field,
                magnetic_field,
                wp.vec3i(*electric_field.shape[1:]),
                spacing,
                dt,
            ],
            dim=num_cells,
        )

        return electric_field, magnetic_field
