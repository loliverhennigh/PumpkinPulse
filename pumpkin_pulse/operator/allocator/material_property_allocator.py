import warp as wp
import numpy as np

from pumpkin_pulse.struct.field import Fielduint8, Fieldfloat32
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
        density_mapping: np.ndarray,
        thermal_conductivity_mapping: np.ndarray,
        solid_fraction_mapping: np.ndarray,
        solid_type_mapping: np.ndarray,
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
        material_properties.id = Fielduint8()
        material_properties.id.data = wp.zeros(shape_with_ghost, dtype=wp.uint8)
        material_properties.id.origin = wp.vec3(origin)
        material_properties.id.spacing = wp.vec3(spacing)
        material_properties.id.shape = wp.vec3i(shape)
        material_properties.id.nr_ghost_cells = nr_ghost_cells

        # Allocate the mc_id information
        material_properties.mc_id = Fielduint8()
        material_properties.mc_id.data = wp.zeros(shape_with_ghost, dtype=wp.uint8)
        material_properties.mc_id.origin = wp.vec3([o + s/2.0 for o, s in zip(origin, spacing)])
        material_properties.mc_id.spacing = wp.vec3(spacing)
        material_properties.mc_id.shape = wp.vec3i(shape)

        # Material information
        # Electrical properties
        material_properties.eps_mapping = wp.from_numpy(eps_mapping, dtype=wp.float32)
        material_properties.mu_mapping = wp.from_numpy(mu_mapping, dtype=wp.float32)
        material_properties.sigma_mapping = wp.from_numpy(sigma_mapping, dtype=wp.float32)

        # Thermal properties
        material_properties.specific_heat_mapping = wp.from_numpy(specific_heat_mapping, dtype=wp.float32)
        material_properties.density_mapping = wp.from_numpy(density_mapping, dtype=wp.float32)
        material_properties.thermal_conductivity_mapping = wp.from_numpy(thermal_conductivity_mapping, dtype=wp.float32)

        # Particle properties
        material_properties.solid_fraction_mapping = wp.from_numpy(solid_fraction_mapping, dtype=wp.uint8)
        material_properties.solid_type_mapping = wp.from_numpy(solid_type_mapping, dtype=wp.uint8)

        return material_properties
