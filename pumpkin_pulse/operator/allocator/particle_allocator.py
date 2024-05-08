import warp as wp

from pumpkin_pulse.struct.particles import Particles, PosMom
from pumpkin_pulse.operator.operator import Operator

class ParticleAllocator(Operator):

    def __call__(
        self,
        nr_particles: wp.int32,
        charge: wp.float32,
        mass: wp.float32,
        origin: wp.vec3,
        spacing: wp.vec3,
        shape: wp.vec3i,
        nr_ghost_cells: wp.int32,
    ):

        # Get the shape with ghost cells
        shape_with_ghost = [s + 2 * nr_ghost_cells for s in shape]

        # Allocate the particle data
        particles = Particles()

        # Allocate the primary particle data
        particles.data = wp.zeros(nr_particles, dtype=PosMom)

        # Allocate the buffers for particle data
        particles.data_buffer = wp.zeros(nr_particles, dtype=PosMom)

        # Set the number of particles
        particles.nr_particles = wp.zeros(1, dtype=wp.int32)

        # Allocate the charge and mass
        particles.charge = wp.float32(charge)
        particles.mass = wp.float32(mass)

        # Allocate the cell particle mapping
        particles.cell_particle_mapping = wp.zeros(shape_with_ghost, dtype=wp.int32)
        particles.cell_particle_mapping_buffer = wp.zeros(shape_with_ghost, dtype=wp.int32)

        # Grid information
        particles.spacing = wp.vec3(spacing)
        particles.shape = wp.vec3i(shape)
        particles.nr_ghost_cells = wp.int32(nr_ghost_cells)

        # Origins for all the fields
        particles.origin = wp.vec3f(origin)
        particles.rho_origin = wp.vec3f([origin[0] - 0.5 * spacing[0], origin[1] - 0.5 * spacing[1], origin[2] - 0.5 * spacing[2]])

        return particles
