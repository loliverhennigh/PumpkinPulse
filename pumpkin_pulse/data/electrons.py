import warp as wp

wp.init()

@wp.struct
class Electrons:
    # Primary particle data
    position: wp.array(dtype=wp.vec3)
    momentum: wp.array(dtype=wp.vec3)

    # Buffers for particle data
    position_buffer: wp.array(dtype=wp.vec3)
    momentum_buffer: wp.array(dtype=wp.vec3)

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
