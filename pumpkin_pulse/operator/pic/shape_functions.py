import numpy as np
import warp as wp
from build123d import Compound
import tempfile
from stl import mesh as np_mesh

from pumpkin_pulse.operator.operator import Operator

@wp.func
def linear_shape_function(
    x: wp.float32,
    x_i: wp.float32,
):
    """
    Linear shape function

    Parameters
    ----------
    x : wp.float32
        Position to evaluate shape function at, Normalized so spacing is 1.0
    x_i : wp.float32
        Position of node, Normalized so spacing is 1.0
    """

    w = 1.0 - wp.abs((x - x_i))
    w = wp.max(w, 0.0)
    return w
