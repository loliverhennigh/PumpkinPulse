# Base structure for particle data

import warp as wp

@wp.struct
class PosMom:
    position: wp.vec3
    momentum: wp.vec3

@wp.struct
class Particles:
    # Primary particle data
    data: wp.array(dtype=PosMom)

    # Buffers for particle data
    data_buffer: wp.array(dtype=PosMom)

    # Particle information
    macro_to_macro_ratio: wp.float32

    # Particle mass and charge
    mass: wp.float32
    charge: wp.float32

    # Number of particles
    nr_particles: wp.array(dtype=wp.int32)

    # Number of particles per grid cell
    cell_particle_mapping: wp.array3d(dtype=wp.int32)
    cell_particle_mapping_buffer: wp.array3d(dtype=wp.int32)

    # Grid information
    spacing: wp.vec3
    shape: wp.vec3i
    nr_ghost_cells: wp.int32

    # Origins for all the fields
    origin: wp.vec3
    rho_origin: wp.vec3

    @wp.func
    def pos_to_cell(
        pos: wp.vec3f,
        origin: wp.vec3f,
        spacing: wp.vec3f,
        nr_ghost_cells: wp.int32,
    ):
        return wp.vec3i(
            wp.int32((pos[0] - origin[0]) / spacing[0]) + nr_ghost_cells,
            wp.int32((pos[1] - origin[1]) / spacing[1]) + nr_ghost_cells,
            wp.int32((pos[2] - origin[2]) / spacing[2]) + nr_ghost_cells,
        )


