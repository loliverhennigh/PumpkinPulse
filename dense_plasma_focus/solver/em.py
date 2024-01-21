# FDTD solver for 3D Yee grid

from dataclasses import dataclass

import warp as wp
import numpy as np

from dense_plasma_focus.material import get_materials_in_compound, Material

wp.init()



@wp.func
def sample_electric_properties(
    material_id: wp.array3d(dtype=wp.uint8),
    material_property: wp.array(dtype=wp.float32),
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
):

    # Get material id for needed cells
    m_0_0_1 = material_id[i - 1, j - 1, k]
    m_0_1_0 = material_id[i - 1, j, k - 1]
    m_0_1_1 = material_id[i - 1, j, k]
    m_1_0_0 = material_id[i, j - 1, k - 1]
    m_1_0_1 = material_id[i, j - 1, k]
    m_1_1_0 = material_id[i, j, k - 1]
    m_1_1_1 = material_id[i, j, k]

    # Get material property
    prop_0_0_1 = material_property[wp.int32(m_0_0_1)]
    prop_0_1_0 = material_property[wp.int32(m_0_1_0)]
    prop_0_1_1 = material_property[wp.int32(m_0_1_1)]
    prop_1_0_0 = material_property[wp.int32(m_1_0_0)]
    prop_1_0_1 = material_property[wp.int32(m_1_0_1)]
    prop_1_1_0 = material_property[wp.int32(m_1_1_0)]
    prop_1_1_1 = material_property[wp.int32(m_1_1_1)]

    #return eps
    return wp.float(m_1_1_1) * eps


def construct_em_update_kernels(
    materials,
    bounds,
):
    """
    Construct kernels for updating electric and magnetic fields
    """

    # Get sample properties function
    sample_electric_property = construct_electric_property_sampler(bounds)

    # Make kernel for updating electric field
    @wp.kernel
    def _update_eletric_field(
        electric_field_x: wp.array3d(dtype=wp.float32),
        electric_field_y: wp.array3d(dtype=wp.float32),
        electric_field_z: wp.array3d(dtype=wp.float32),
        magnetic_field_x: wp.array3d(dtype=wp.float32),
        magnetic_field_y: wp.array3d(dtype=wp.float32),
        magnetic_field_z: wp.array3d(dtype=wp.float32),
        impressed_electric_current_x: wp.array3d(dtype=wp.vec3),
        impressed_electric_current_y: wp.array3d(dtype=wp.vec3),
        impressed_electric_current_z: wp.array3d(dtype=wp.vec3),
        material_id: wp.array3d(dtype=wp.uint8),
        eps_mapping: wp.array(dtype=wp.float32),
        sigma_e_mapping: wp.array(dtype=wp.float32),
        dx: float,
        dt: float,
    ):
    
        # get index
        i, j, k = wp.tid()

        # get properties
        eps = sample_electric_propery(material_id, eps_mapping, i, j, k)

        # Set electric field to zero if in bounds
        electric_field[i, j, k] = wp.float(eps) * wp.vec3(1.0, 1.0, 1.0)

    return _update_eletric_field



if __name__ == "__main__":
    # imports
    from dense_plasma_focus.reactor.reactor import LLPReactor
    from dense_plasma_focus.solver.voxelizer import voxelize_compound

    # Create reactor
    reactor = LLPReactor()

    # Get bounding box
    bounding_box = reactor.bounding_box()
    origin = (bounding_box.min.X, bounding_box.min.Y, bounding_box.min.Z)
    dx = 0.5
    nr_voxels = (
        int((bounding_box.max.X - bounding_box.min.X) / dx),
        int((bounding_box.max.Y - bounding_box.min.Y) / dx),
        int((bounding_box.max.Z - bounding_box.min.Z) / dx),
    )

    # Voxelize electrode
    materials = get_materials_in_compound(reactor)
    material_id = voxelize_compound(
        compound=reactor,
        origin=origin,
        resolution=dx,
        nr_voxels=nr_voxels,
        materials=materials,
    )

    # Make electric field
    electric_field_x = wp.zeros(
        (nr_voxels[0], nr_voxels[1]+1, nr_voxels[2]+1), wp.float32
    )
    electric_field_y = wp.zeros(
        (nr_voxels[0]+1, nr_voxels[1], nr_voxels[2]+1), wp.float32
    )
    electric_field_z = wp.zeros(
        (nr_voxels[0]+1, nr_voxels[1]+1, nr_voxels[2]), wp.float32
    )

    # Make magnetic field
    magnetic_field_x = wp.zeros(
        (nr_voxels[0]+1, nr_voxels[1], nr_voxels[2]), wp.float32
    )
    magnetic_field_y = wp.zeros(
        (nr_voxels[0], nr_voxels[1]+1, nr_voxels[2]), wp.float32
    )
    magnetic_field_z = wp.zeros(
        (nr_voxels[0], nr_voxels[1], nr_voxels[2]+1), wp.float32
    )

    # Get em kernels
    update_electric_field_kernel = construct_em_update_kernels(materials, None)

    # Run kernel
    wp.launch(
        update_electric_field_kernel,
        inputs=[
            electric_field,
            magnetic_field,
            impressed_electric_current,
            material_id,
            eps_mapping,
            1.0,
            1.0,
        ],
        dim=(10, 10, 10),
    )
