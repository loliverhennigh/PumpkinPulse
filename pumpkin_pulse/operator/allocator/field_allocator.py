import warp as wp
import numpy as np

from pumpkin_pulse.struct.field import Field
from pumpkin_pulse.operator.operator import Operator

class FieldAllocator(Operator):

    def __call__(
        self,
        dtype: wp.int32,
        origin: wp.vec3,
        spacing: wp.vec3,
        shape: wp.vec3i,
        nr_ghost_cells: wp.int32,
    ):

        # Get the shape with ghost cells
        shape_with_ghost = [s + 2 * nr_ghost_cells for s in shape]

        # Allocate the field
        field = Field(dtype=dtype)

        # Grid information
        field.shape = wp.vec3i(shape)
        field.spacing = wp.vec3(spacing)
        field.origin = wp.vec3(origin)
        field.nr_ghost_cells = nr_ghost_cells

        return field
