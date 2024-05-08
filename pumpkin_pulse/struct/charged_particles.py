# Neutral particles data structure

import warp as wp

from pumpkin_pulse.struct.particles import Particles, PosMom


@wp.struct
#class ChargedParticles(Particles): # TODO: warp inheritance from wp.struct does not work yet
class ChargedParticles:
    # Primary particle data
    data: wp.array(dtype=PosMom)

    # Buffers for particle data
    data_buffer: wp.array(dtype=PosMom)

    # Particle information
    macro_to_macro_ratio: wp.float32

    # Particle mass, charge and atomic number
    mass: wp.float32
    charge: wp.float32
    atomic_number: wp.float32

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
