# Base class for voxelizing geometries
import numpy as np
import warp as wp

from pumpkin_pulse.data.field import Fielduint8
from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.operator.voxelize.sdf import SignedDistanceFunction
from pumpkin_pulse.operator.voxelize.tube import Tube

class Geometry(Operator):
    """
    Base class for all geometries
    """

    @staticmethod
    def _rotate_point(
        point: np.ndarray,
        center: np.ndarray,
        centeraxis: np.ndarray,
        angle: float,
    ):
        """
        Rotate a point around an axis
        """

        # Normalize centeraxis
        centeraxis = centeraxis / np.linalg.norm(centeraxis)

        # Create rotation matrix
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        x = centeraxis[0]
        y = centeraxis[1]
        z = centeraxis[2]
        rotation_matrix = np.array(
            [
                [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
                [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
                [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
            ]
        )

        # Translate point to origin
        point = point - center

        # Rotate point
        point = np.dot(rotation_matrix, point)

        # Translate point back
        point = point + center

        return point
