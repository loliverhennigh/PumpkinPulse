# Base class for voxelizing geometries
import warp as wp

from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.struct.field import Fielduint8
from pumpkin_pulse.functional.marching_cube import get_mc_id

class IDFieldToMCField(Operator):
    """
    Operator to set the marching cube id field from the id field
    """

    @wp.kernel 
    def _id_field_to_mc_field(
        id_field: Fielduint8,
        solid_mappping: wp.array(dtype=wp.uint8),
        mc_field: Fielduint8,
    ):
        i, j, k = wp.tid()

        # Get stencil of id
        s_000 = solid_mappping[wp.int32(id_field.data[i, j, k])]
        s_100 = solid_mappping[wp.int32(id_field.data[i+1, j, k])]
        s_010 = solid_mappping[wp.int32(id_field.data[i, j+1, k])]
        s_110 = solid_mappping[wp.int32(id_field.data[i+1, j+1, k])]
        s_001 = solid_mappping[wp.int32(id_field.data[i, j, k+1])]
        s_101 = solid_mappping[wp.int32(id_field.data[i+1, j, k+1])]
        s_011 = solid_mappping[wp.int32(id_field.data[i, j+1, k+1])]
        s_111 = solid_mappping[wp.int32(id_field.data[i+1, j+1, k+1])]

        # Get marching cube id
        mc_id = get_mc_id(s_000, s_100, s_010, s_110, s_001, s_101, s_011, s_111)

        # set marching cube id to mc_field
        mc_field.data[i, j, k] = mc_id

    def __call__(
        self,
        id_field: Fielduint8,
        solid_mappping: wp.array,
        mc_field: Fielduint8,
    ):
        # Voxelize STL of mesh
        wp.launch(
            self._id_field_to_mc_field,
            inputs=[
                id_field,
                solid_mappping,
                mc_field,
            ],
            dim=mc_field.shape,
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
