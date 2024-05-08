import warp as wp

@wp.struct
class MaterialProperties:
    # Id information
    id: wp.array3d(dtype=wp.uint8)

    # Temperature information
    temperature: wp.array3d(dtype=wp.float32)

    # Material information mappings
    # Electrical properties
    eps_mapping: wp.array(dtype=wp.float32)
    mu_mapping: wp.array(dtype=wp.float32)
    sigma_mapping: wp.array(dtype=wp.float32)

    # Thermal properties
    specific_heat_mapping: wp.array(dtype=wp.float32)
    density_mapping: wp.array(dtype=wp.float32)
    thermal_conductivity_mapping: wp.array(dtype=wp.float32)

    # Particle properties
    solid_mapping: wp.array(dtype=wp.float32)
    kind_mapping: wp.array(dtype=wp.uint8)

    # Grid information
    origin: wp.vec3
    spacing: wp.vec3
    shape: wp.vec3i
    nr_ghost_cells: wp.int32
