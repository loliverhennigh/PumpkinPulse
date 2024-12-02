# Base class for voxelizing geometries
import numpy as np
import warp as wp

from pumpkin_pulse.data.field import Fielduint8
from pumpkin_pulse.operator.geometry.geometry import Geometry

class Circuit(Geometry):
    """
    Base class for all circular geometries
    """

    def __init__(
        self,
        input_point: np.ndarray,
        input_normal: np.ndarray,
        output_point: np.ndarray,
        output_normal: np.ndarray,
    ):
        super().__init__()
        self.input_point = input_point
        self.input_normal = input_normal
        self.output_point = output_point
        self.output_normal = output_normal
