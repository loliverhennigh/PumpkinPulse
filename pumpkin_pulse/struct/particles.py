import warp as wp

from pumpkin_pulse.struct.field import Fieldint32

@wp.struct
class Particles:
    # Primary particle data
    position: wp.array(dtype=wp.vec3)
    momentum: wp.array(dtype=wp.vec3)
    kill: wp.array(dtype=wp.uint8)

    # Buffers for particle data
    position_buffer: wp.array(dtype=wp.vec3)
    momentum_buffer: wp.array(dtype=wp.vec3)
    kill_buffer: wp.array(dtype=wp.uint8)

    # Particle information
    weighting: wp.float32

    # Particle mass and charge
    mass: wp.float32
    charge: wp.float32

    # Number of particles
    nr_particles: wp.array(dtype=wp.int32)

    # Number of particles per grid cell
    cell_particle_mapping: Fieldint32
    cell_particle_mapping_buffer: Fieldint32
