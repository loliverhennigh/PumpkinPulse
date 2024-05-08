# Base class for voxelizing geometries
import warp as wp

from pumpkin_pulse.operator.operator import Operator

class Voxelize(Operator):
    """
    Base class for voxelizing geometries
    """

    @wp.kernel
    def _voxelize_mesh(
        voxel_id: wp.array4d(dtype=wp.uint8),
        mesh: wp.uint64,
        id_number: int,
        spacing: wp.vec3,
        origin: wp.vec3,
        nr_ghost_cells: int,
    ):
    
        # get spatial index
        i, j, k = wp.tid()

        # Get index of voxel offset by ghost cells
        voxel_i = i + nr_ghost_cells
        voxel_j = j + nr_ghost_cells
        voxel_k = k + nr_ghost_cells

        # Offset by ghost cells
        i += nr_ghost_cells
        j += nr_ghost_cells
        k += nr_ghost_cells
    
        # position of voxel
        ijk = wp.vec3(wp.float(i), wp.float(j), wp.float(k))
        ijk = ijk + wp.vec3(0.5, 0.5, 0.5) # cell center
        pos = wp.cw_mul(ijk, spacing) + origin

        # Only evaluate voxel if not set yet
        if voxel_id[0, voxel_i, voxel_j, voxel_k] != wp.uint8(0):
            return

        # Compute maximum distance to check
        max_length = wp.sqrt(
             (spacing[0] * wp.float32(voxel_id.shape[1]))**2.0
             + (spacing[1] * wp.float32(voxel_id.shape[2]))**2.0
             + (spacing[2] * wp.float32(voxel_id.shape[3]))**2.0
        )
    
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
                voxel_id[0, voxel_i, voxel_j, voxel_k] = wp.uint8(255)
            elif (sign < 0): # TODO: fix this
                voxel_id[0, voxel_i, voxel_j, voxel_k] = wp.uint8(id_number) 
            else:
                pass
