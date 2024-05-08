# Base class for voxelizing geometries
import warp as wp

from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.struct.material_properties import MaterialProperties

class MeshMaterialPropertySetter(Operator):
    """
    Operator to set the material properties of a mesh
    """

    @wp.kernel
    def _set_mesh(
        material_properties: MaterialProperties,
        id_number: int,
        mesh: wp.uint64,
    ):
    
        # get spatial index
        i, j, k = wp.tid()

        # Get voxel index
        voxel_i = i + material_properties.nr_ghost_cells
        voxel_j = j + material_properties.nr_ghost_cells
        voxel_k = k + material_properties.nr_ghost_cells

        # position of voxel (cell center)
        ijk = wp.vec3(wp.float(i), wp.float(j), wp.float(k))
        ijk = ijk + wp.vec3(0.5, 0.5, 0.5) # cell center
        pos = wp.cw_mul(ijk, material_properties.spacing) + material_properties.origin

        # Compute maximum distance to check
        max_length = wp.sqrt(
                (material_properties.spacing[0] * wp.float32(material_properties.shape[1]))**2.0
                + (material_properties.spacing[1] * wp.float32(material_properties.shape[2]))**2.0
                + (material_properties.spacing[2] * wp.float32(material_properties.shape[3]))**2.0
        )
    
        # evaluate distance of point
        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)
        if (wp.mesh_query_point_sign_winding_number(mesh, pos, max_length, sign, face_index, face_u, face_v)):

            # set point to be solid
            if sign < 0.0:
                material_properties.id[voxel_i, voxel_j, voxel_k] = wp.uint8(id_number)

    def __call__(
        self,
        material_properties: MaterialProperties,
        mesh: wp.Mesh,
        id_number: int,
        tolerance: float = 0.001,
        angular_tolerance: float = 0.1,
    ):
        # Voxelize STL of mesh
        wp.launch(
            self._set_mesh,
            inputs=[
                material_properties,
                id_number,
                mesh.id,
            ],
            dim=[x - 2 * material_properties.nr_ghost_cells for x in material_properties.shape],
        )

        return material_properties
