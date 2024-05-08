import warp as wp
import numpy as np

from pumpkin_pulse.struct.material_properties import MaterialProperties
from pumpkin_pulse.operator.operator import Operator

class MaterialPropertyAllocator(Operator):

    def __call__(
        self,
        nr_materials: wp.int32,
        eps_mapping: np.ndarray,
        mu_mapping: np.ndarray,
        sigma_mapping: np.ndarray,
        specific_heat_mapping: np.ndarray,
        solid_mapping: np.ndarray,
        origin: wp.vec3,
        spacing: wp.vec3,
        shape: wp.vec3i,
        nr_ghost_cells: wp.int32,
    ):

        # Get the shape with ghost cells
        shape_with_ghost = [s + 2 * nr_ghost_cells for s in shape]

        # Allocate the MaterialProperties object
        material_properties = MaterialProperties()

        # Allocate the id information
        material_properties.id = wp.zeros(shape_with_ghost, dtype=wp.uint8)

        # Material information
        material_properties.eps_mapping = wp.from_numpy(eps_mapping, dtype=wp.float32)
        material_properties.mu_mapping = wp.from_numpy(mu_mapping, dtype=wp.float32)
        material_properties.sigma_mapping = wp.from_numpy(sigma_mapping, dtype=wp.float32)
        material_properties.specific_heat_mapping = wp.from_numpy(specific_heat_mapping, dtype=wp.float32)
        material_properties.solid_mapping = wp.from_numpy(solid_mapping, dtype=wp.float32)

        # Grid information
        material_properties.shape = wp.vec3i(shape_with_ghost)
        material_properties.spacing = wp.vec3(spacing)
        material_properties.origin = wp.vec3(origin)
        material_properties.nr_ghost_cells = nr_ghost_cells

        return material_properties
