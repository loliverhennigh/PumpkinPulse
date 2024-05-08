import numpy as np
import warp as wp
from build123d import Compound
import tempfile
from stl import mesh as np_mesh

from pumpkin_pulse.operator.operator import Operator

def feild_to_particle_factory(
    shape_function: wp.func,
    ):
    """
    Factory function to create a function that interpolates a feild to a particle
    using a given shape function.

    Parameters
    ----------
    shape_function : wp.func
        Shape function to use for interpolation
            
    Returns
    -------
    wp.func
        Function that interpolates a feild to a particle
    """

    @wp.func
    def feild_to_particle(
        pos: wp.vec3f,
        ijk_lower: wp.vec3i,
        feild: wp.array3d(dtype=wp.float32),
    ) -> wp.float32:
        """
        Interpolates a feild to a particle

        Parameters
        ----------
        pos : wp.vec3f
            Position of particle, normalized so spacing is 1.0
        ijk_lower : wp.vec3i
            Lower corner of the cell that the particle is in
        feild : wp.array3d(dtype=wp.float32)
            Feild to interpolate from

        Returns
        -------
        wp.float32
            Interpolated feild value
        """
    
        # Get feild values at corners
        f_0_0_0 = feild[ijk_lower[0], ijk_lower[1], ijk_lower[2]]
        f_1_0_0 = feild[ijk_lower[0] + 1, ijk_lower[1], ijk_lower[2]]
        f_0_1_0 = feild[ijk_lower[0], ijk_lower[1] + 1, ijk_lower[2]]
        f_1_1_0 = feild[ijk_lower[0] + 1, ijk_lower[1] + 1, ijk_lower[2]]
        f_0_0_1 = feild[ijk_lower[0], ijk_lower[1], ijk_lower[2] + 1]
        f_1_0_1 = feild[ijk_lower[0] + 1, ijk_lower[1], ijk_lower[2] + 1]
        f_0_1_1 = feild[ijk_lower[0], ijk_lower[1] + 1, ijk_lower[2] + 1]
        f_1_1_1 = feild[ijk_lower[0] + 1, ijk_lower[1] + 1, ijk_lower[2] + 1]
    
        # Get shape functions for each corner
        s_x_lower = shape_function(pos[0], 0.0)
        s_y_lower = shape_function(pos[1], 0.0)
        s_z_lower = shape_function(pos[2], 0.0)
        s_x_upper = shape_function(pos[0], 1.0)
        s_y_upper = shape_function(pos[1], 1.0)
        s_z_upper = shape_function(pos[2], 1.0)
    
        # Interpolate feild
        f = (
            f_0_0_0 * s_x_lower * s_y_lower * s_z_lower
            + f_1_0_0 * s_x_upper * s_y_lower * s_z_lower
            + f_0_1_0 * s_x_lower * s_y_upper * s_z_lower
            + f_1_1_0 * s_x_upper * s_y_upper * s_z_lower
            + f_0_0_1 * s_x_lower * s_y_lower * s_z_upper
            + f_1_0_1 * s_x_upper * s_y_lower * s_z_upper
            + f_0_1_1 * s_x_lower * s_y_upper * s_z_upper
            + f_1_1_1 * s_x_upper * s_y_upper * s_z_upper
        )
    
        return f

    return feild_to_particle

