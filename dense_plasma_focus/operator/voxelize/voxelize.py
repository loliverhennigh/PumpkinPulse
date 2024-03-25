# Base class for voxelizing geometries

from typing import Union
import warp as wp

from dense_plasma_focus.operator.operator import Operator
from dense_plasma_focus.compute_backend import ComputeBackend
from dense_plasma_focus.material import Material

class Voxelize(Operator):
    """
    Base class for voxelizing geometries
    """

    @wp.kernel
    def _voxelize_mesh(
        voxels: wp.array3d(dtype=wp.uint8),
        mesh: wp.uint64,
        spacing: wp.vec3,
        origin: wp.vec3,
        shape: wp.vec(3, wp.uint32),
        max_length: float,
        material_id: int
    ):
    
        # get index of voxel
        i, j, k = wp.tid()
    
        # position of voxel
        ijk = wp.vec3(wp.float(i), wp.float(j), wp.float(k))
        ijk = ijk + wp.vec3(0.5, 0.5, 0.5) # cell center
        pos = wp.cw_mul(ijk, spacing) + origin

        # Only evaluate voxel if not set yet
        if voxels[i, j, k] != wp.uint8(0):
            return
    
        # evaluate distance of point
        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)
        if (wp.mesh_query_point(mesh, pos, max_length, sign, face_index, face_u, face_v)):
            p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
            delta = pos - p
            norm = wp.sqrt(wp.dot(delta, delta))
    
            # set point to be solid
            if norm < wp.min(spacing):
                voxels[i, j, k] = wp.uint8(255)
            elif (sign < 0): # TODO: fix this
                voxels[i, j, k] = wp.uint8(material_id)
            else:
                pass
