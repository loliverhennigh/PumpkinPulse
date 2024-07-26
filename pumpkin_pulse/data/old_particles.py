import warp as wp

from pumpkin_pulse.struct.field import Fieldint32

@wp.struct
class Particle:
    position: wp.vec3
    momentum: wp.vec3
    kill: wp.uint8 # 0 = alive, 1 = dead

@wp.struct
class Particles:
    # Primary particle data
    data: wp.array(dtype=Particle)

    # Buffers for particle data
    data_buffer: wp.array(dtype=Particle)

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
