# Base class for voxelizing geometries
import warp as wp

from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.struct.material_properties import MaterialProperties

class SDFMaterialPropertySetter(Operator):
    """
    Base class for setting material properties based on a signed distance function
    """

    def __init__(
        self,
        sdf_func,
    ):

        # Set the sdf function
        self.sdf_func = sdf_func

        @wp.kernel
        def _set_material_properties(
            material_properties: MaterialProperties,
            id_number: int,
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

            # Set material properties
            if self.sdf_func(pos) < 0.0:
                material_properties.id[voxel_i, voxel_j, voxel_k] = wp.uint8(id_number)

        self._set_material_properties = _set_material_properties

    def __call__(
        self,
        material_properties: MaterialProperties,
        id_number: int,
    ):

        # Launch kernel
        wp.launch(
            self._set_material_properties,
            inputs=[
                material_properties,
                id_number,
            ],
            dim=[x - 2 * material_properties.nr_ghost_cells for x in material_properties.shape],
        )

        return material_properties

class SphereMaterialPropertySetter(SDFMaterialPropertySetter):
    """
    Class for setting material properties based on a sphere
    """

    def __init__(
        self,
        center,
        radius,
    ):

        center = wp.constant(wp.vec3(*center))
        radius = wp.constant(wp.float32(radius))

        # Set the sdf function
        @wp.func
        def sphere_sdf_func(pos: wp.vec3):
            return wp.length(pos - center) - radius

        super().__init__(sphere_sdf_func)

# TODO: Warp function signatures break
class BoxMaterialPropertySetter(SDFMaterialPropertySetter):
    """
    Class for setting material properties based on a box
    """

    def __init__(
        self,
        center,
        size,
    ):

        center = wp.constant(wp.vec3(*center))
        size = wp.constant(wp.vec3(*size))

        # Set the sdf function
        @wp.func
        def box_sdf_func(pos: wp.vec3):
            dx = wp.abs(pos[0] - center[0]) - size[0] / 2.0
            dy = wp.abs(pos[1] - center[1]) - size[1] / 2.0
            dz = wp.abs(pos[2] - center[2]) - size[2] / 2.0
            outside = wp.sqrt(dx * dx + dy * dy + dz * dz)
            inside = wp.min(wp.max(pos[0] - size[0] / 2.0, wp.max(pos[1] - size[1] / 2.0, pos[2] - size[2] / 2.0)), 0.0)
            return outside + inside

        super().__init__(box_sdf_func)
