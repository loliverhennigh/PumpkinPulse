# Base class for voxelizing geometries
import warp as wp

from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.struct.material_properties import MaterialProperties
from pumpkin_pulse.functional.marching_cube import get_mc_id

class MeshMaterialPropertySetter(Operator):
    """
    Operator to set the material properties of a mesh
    """

    @wp.kernel
    def _set_id_from_mesh(
        material_properties: MaterialProperties,
        id_number: int,
        mesh: wp.uint64,
    ):
    
        # get spatial index
        i, j, k = wp.tid()

        # Get voxel index
        voxel_i = i + material_properties.id.nr_ghost_cells
        voxel_j = j + material_properties.id.nr_ghost_cells
        voxel_k = k + material_properties.id.nr_ghost_cells

        # position of voxel (cell center)
        ijk = wp.vec3(wp.float(i), wp.float(j), wp.float(k))
        ijk = ijk + wp.vec3(0.5, 0.5, 0.5) # cell center
        pos = wp.cw_mul(ijk, material_properties.id.spacing) + material_properties.id.origin

        # Compute maximum distance to check
        max_length = wp.sqrt(
                (material_properties.id.spacing[0] * wp.float32(material_properties.id.shape[1]))**2.0
                + (material_properties.id.spacing[1] * wp.float32(material_properties.id.shape[2]))**2.0
                + (material_properties.id.spacing[2] * wp.float32(material_properties.id.shape[3]))**2.0
        )
    
        # evaluate distance of point
        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)
        if (wp.mesh_query_point_sign_winding_number(mesh, pos, max_length, sign, face_index, face_u, face_v)):

            # set point to be solid
            if sign < 0.0:
                material_properties.id.data[voxel_i, voxel_j, voxel_k] = wp.uint8(id_number)

    @wp.kernel 
    def _update_mc_id_from_id(
        material_properties: MaterialProperties,
    ):
        i, j, k = wp.tid()

        # Get stencil of id
        id_000 = material_properties.id.data[i, j, k]
        id_100 = material_properties.id.data[i+1, j, k]
        id_010 = material_properties.id.data[i, j+1, k]
        id_110 = material_properties.id.data[i+1, j+1, k]
        id_001 = material_properties.id.data[i, j, k+1]
        id_101 = material_properties.id.data[i+1, j, k+1]
        id_011 = material_properties.id.data[i, j+1, k+1]
        id_111 = material_properties.id.data[i+1, j+1, k+1]

        # Get mc_id
        mc_id = get_mc_id(id_000, id_100, id_010, id_110, id_001, id_101, id_011, id_111)

        # set mc_id to id
        material_properties.mc_id.data[i, j, k] = mc_id

    def __call__(
        self,
        material_properties: MaterialProperties,
        mesh: wp.Mesh,
        id_number: int,
    ):
        # Voxelize STL of mesh
        wp.launch(
            self._set_id_from_mesh,
            inputs=[
                material_properties,
                id_number,
                mesh.id,
            ],
            dim=material_properties.id.shape,
        )

        # Update mc_id
        wp.launch(
            self._update_mc_id_from_id,
            inputs=[
                material_properties,
            ],
            dim=material_properties.mc_id.shape,
        )

        return material_properties
