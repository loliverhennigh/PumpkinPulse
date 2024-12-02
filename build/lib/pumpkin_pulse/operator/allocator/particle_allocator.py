import warp as wp

from pumpkin_pulse.struct.field import Fieldint32
from pumpkin_pulse.struct.particles import Particles
from pumpkin_pulse.operator.operator import Operator

class ParticleAllocator(Operator):

    def __call__(
        self,
        nr_particles: wp.int32,
        weight: wp.float32,
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

        # Allocate the particle data and buffer
        particles.position = wp.zeros(nr_particles, dtype=wp.vec3f)
        particles.momentum = wp.zeros(nr_particles, dtype=wp.vec3f)
        particles.kill = wp.zeros(nr_particles, dtype=wp.uint8)
        particles.position_buffer = wp.zeros(nr_particles, dtype=wp.vec3f)
        particles.momentum_buffer = wp.zeros(nr_particles, dtype=wp.vec3f)
        particles.kill_buffer = wp.zeros(nr_particles, dtype=wp.uint8)

        # Set the number of particles
        particles.nr_particles = wp.zeros(1, dtype=wp.int32)

        # Allocate the weighting, charge, and mass
        particles.weighting = wp.float32(weight)
        particles.charge = wp.float32(charge)
        particles.mass = wp.float32(mass)

        # Allocate the cell particle mapping
        particles.cell_particle_mapping = Fieldint32()
        particles.cell_particle_mapping.data = wp.zeros(shape_with_ghost, dtype=wp.int32)
        particles.cell_particle_mapping.origin = wp.vec3f([o - 0.5 * s for o, s in zip(origin, spacing)])
        particles.cell_particle_mapping.spacing = wp.vec3f(spacing)
        particles.cell_particle_mapping.shape = wp.vec3i(shape)
        particles.cell_particle_mapping.nr_ghost_cells = wp.int32(nr_ghost_cells)
        particles.cell_particle_mapping_buffer = Fieldint32()
        particles.cell_particle_mapping_buffer.data = wp.zeros(shape_with_ghost, dtype=wp.int32)
        particles.cell_particle_mapping_buffer.origin = wp.vec3f([o - 0.5 * s for o, s in zip(origin, spacing)])
        particles.cell_particle_mapping_buffer.spacing = wp.vec3f(spacing)
        particles.cell_particle_mapping_buffer.shape = wp.vec3i(shape)
        particles.cell_particle_mapping_buffer.nr_ghost_cells = wp.int32(nr_ghost_cells)

        return particles
