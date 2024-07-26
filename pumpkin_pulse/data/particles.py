import warp as wp

from pumpkin_pulse.data.field import Fieldint32

@wp.struct
class Particles:
    # Primary particle data
    position: wp.array(dtype=wp.vec3)
    momentum: wp.array(dtype=wp.vec3)
    weight: wp.array(dtype=wp.float32)
    kill: wp.array(dtype=wp.uint8)

    # Number of particles
    num_particles: wp.array(dtype=wp.int32)

    # Particle mass and charge (unweighted)
    mass: wp.float32
    charge: wp.float32
    volume: wp.float32

    # Grid indexing
    grid_index: Fieldint32
