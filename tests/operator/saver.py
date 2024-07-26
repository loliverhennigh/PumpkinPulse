import pytest

import warp as wp
import numpy as np

from pumpkin_pulse.operator.allocator import FieldAllocator
from pumpkin_pulse.operator.saver import FieldSaver

def test_field_saver():
    # Make operators
    field_allocator = FieldAllocator()
    field_saver = FieldSaver()

    # Allocate field
    field = field_allocator(
        dtype=wp.float32,
        origin=(0, 0, 0),
        spacing=(1, 1, 1),
        shape=(10, 10, 10),
        nr_ghost_cells=1,
    )

    # Make fake np for testing
    data = np.zeros((12, 12, 12), dtype=np.float32)
    data[4:8, 4:8, 4:8] = 1.0
    field.data = wp.from_numpy(data, wp.float32)

    # Save field
    field_saver(field, "test_field_saver.vtk")
