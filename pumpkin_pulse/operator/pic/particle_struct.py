
import warp as wp

from pumpkin_pulse.operator.operator import Operator

wp.init()

@wp.struct
class Particles:
    # Primary particle data
    position: wp.array(dtype=wp.vec3)
    momentum: wp.array(dtype=wp.vec3)
    id: wp.array(dtype=wp.uint8)

    # Buffers for particle data
    position_buffer: wp.array(dtype=wp.vec3)
    momentum_buffer: wp.array(dtype=wp.vec3)
    id_buffer: wp.array(dtype=wp.uint8)

    # Mapping for particle data
    sort_mapping: wp.array(dtype=wp.int32)

    # Particle mass and charge mappings
    mass_mapping: wp.array(dtype=wp.float32)
    charge_mapping: wp.array(dtype=wp.float32)

    # Number of particles
    nr_particles: wp.int32

    # Hash table for particle sorting
    hash_table: wp.array3d(dtype=wp.int32)

    # Grid information
    spacing: wp.vec3
    shape: wp.vec3i
    nr_ghost_cells: wp.int32

    # Origins for all the fields
    origin: wp.vec3
    rho_origin: wp.vec3


class AllocateParticles(Operator):

    def __call__(
        self,
        nr_particles: wp.int32,
        origin: wp.vec3,
        spacing: wp.vec3,
        shape: wp.vec3i,
        nr_id_types: wp.int32,
        nr_ghost_cells: wp.int32
    ):

        # Get the shape with ghost cells
        shape_with_ghost = [s + 2 * nr_ghost_cells for s in shape]

        # Allocate the particle data
        particles = Particles()

        # Allocate the primary particle data 
        particles.position = wp.zeros(nr_particles, dtype=wp.vec3)
        particles.momentum = wp.zeros(nr_particles, dtype=wp.vec3)
        particles.id = wp.zeros(nr_particles, dtype=wp.uint8)

        # Allocate the buffers for particle data
        particles.position_buffer = wp.zeros(nr_particles, dtype=wp.vec3)
        particles.momentum_buffer = wp.zeros(nr_particles, dtype=wp.vec3)
        particles.id_buffer = wp.zeros(nr_particles, dtype=wp.uint8)

        # Allocate the mapping for particle data
        particles.sort_mapping = wp.zeros(nr_particles, dtype=wp.int32)

        # Allocate the particle mass and charge mappings
        particles.mass_mapping = wp.zeros(nr_id_types, dtype=wp.float32)
        particles.charge_mapping = wp.zeros(nr_id_types, dtype=wp.float32)

        # Set the number of particles
        particles.nr_particles = 0

        # Allocate the hash table for particle sorting
        particles.hash_table = wp.zeros(shape_with_ghost, dtype=wp.int32)

        # Grid information
        particles.spacing = wp.vec3(spacing)
        particles.shape = wp.vec3i(shape)
        particles.nr_ghost_cells = nr_ghost_cells

        # Origins for all the fields
        particles.origin = wp.vec3(origin)
        particles.rho_origin = wp.vec3([origin[0] - 0.5 * spacing[0], origin[1] - 0.5 * spacing[1], origin[2] - 0.5 * spacing[2]])

        return particles

if __name__ == '__main__':
    # Allocate the particle data
    allocate_particles = AllocateParticles()
    particles = allocate_particles(
        nr_particles=1000,
        origin=[0.0, 0.0, 0.0],
        spacing=[0.1, 0.1, 0.1],
        shape=[10, 10, 10],
        nr_id_types=2,
        nr_ghost_cells=1
    )

    # Print the allocated particle data
    print(particles)
