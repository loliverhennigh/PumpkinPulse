import warp as wp
import numpy as np

from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.operator.sort.sort import ParticleSorter
from pumpkin_pulse.struct.particles import Particles

class ParticleInjector(Operator):
    _k = wp.constant(1.380649e-23) # Boltzmann constant
    part_sorter = ParticleSorter()

    @wp.func
    def _get_nr_particles_in_cell(
        cell_particle_mapping: wp.array3d(dtype=wp.int32),
        ijk: wp.vec3i,
    ) -> wp.int32:

        # Get next cell index, 
        ordered_i = (
            cell_particle_mapping.shape[1] * cell_particle_mapping.shape[2] * ijk[0]
            + cell_particle_mapping.shape[2] * ijk[1]
            + ijk[2]
        )

        # Get next index
        ordered_i += 1

        # Convert to 3D index
        k = ordered_i % cell_particle_mapping.shape[2]
        j = ((ordered_i - k) / cell_particle_mapping.shape[2]) % cell_particle_mapping.shape[1]
        i = ((ordered_i - (j * cell_particle_mapping.shape[2]) - k) / (cell_particle_mapping.shape[1] * cell_particle_mapping.shape[2]))

        return cell_particle_mapping[i, j, k] - cell_particle_mapping[ijk[0], ijk[1], ijk[2]]

    @wp.kernel
    def _add_particles(
        particles: Particles,
        mesh: wp.uint64,
        nr_particles_per_cell: wp.int32,
        temperature: wp.float32,
        mean_velocity: wp.vec3f,
        seed: wp.int32,
    ):

        # Get cell index
        i, j, k = wp.tid()

        # Set cell particle mapping buffer to store number of particles in cell
        nr_particles_in_cell = ParticleInjector._get_nr_particles_in_cell(particles.cell_particle_mapping.data, wp.vec3i(i, j, k))
        particles.cell_particle_mapping_buffer.data[i, j, k] = nr_particles_in_cell

        # Compute maximum distance to check
        max_distance = wp.sqrt(
            (particles.cell_particle_mapping.spacing[0] * wp.float32(particles.cell_particle_mapping.shape[0])) ** 2.0
            + (particles.cell_particle_mapping.spacing[1] * wp.float32(particles.cell_particle_mapping.shape[1])) ** 2.0
            + (particles.cell_particle_mapping.spacing[2] * wp.float32(particles.cell_particle_mapping.shape[2])) ** 2.0
        )

        # Get center of cell
        ijk = wp.vec3f(
            wp.float32(i),
            wp.float32(j),
            wp.float32(k),
        )
        pos = particles.cell_particle_mapping.origin + wp.cw_mul(particles.cell_particle_mapping.spacing, ijk) + 0.5 * particles.cell_particle_mapping.spacing

        # Check if cell is inside the mesh
        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)
        if (wp.mesh_query_point(mesh, pos, max_distance, sign, face_index, face_u, face_v)):
           if (sign < 0):

               # Get lower and upper bounds of cell
               lower_bound = particles.cell_particle_mapping.origin + wp.cw_mul(particles.cell_particle_mapping.spacing, ijk)
               upper_bound = particles.cell_particle_mapping.origin + wp.cw_mul(particles.cell_particle_mapping.spacing, ijk + wp.vec3f(1.0, 1.0, 1.0))

               # Initialize random seed
               r = wp.rand_init(seed, i * particles.cell_particle_mapping.shape[1] * particles.cell_particle_mapping.shape[2] + j * particles.cell_particle_mapping.shape[2] + k)

               # Get index of last particles in particle list
               start_index = wp.atomic_add(particles.nr_particles, 0, nr_particles_per_cell)

               # Add particles to end of particle list
               for p in range(nr_particles_per_cell):

                   # Get random position in cell
                   particle_pos = lower_bound + wp.cw_mul((upper_bound - lower_bound), wp.vec3f(wp.randf(r), wp.randf(r), wp.randf(r)))

                   # Get mass of particle
                   mass = particles.mass

                   # Get sigma for random velocity
                   sigma = wp.sqrt(ParticleInjector._k * temperature / mass)

                   # Get random velocity TODO: fix normal inf issue
                   #vel = wp.vec3f(sigma * wp.randn(r), sigma * wp.randn(r), sigma * wp.randn(r))
                   vel = wp.vec3f(sigma * (wp.randf(r) - 0.5), sigma * (wp.randf(r) - 0.5), sigma * (wp.randf(r) - 0.5))

                   # Add mean velocity
                   vel += mean_velocity

                   # Set particle data
                   particles.position[start_index + p] = particle_pos
                   particles.momentum[start_index + p] = mass * vel
                   particles.kill[start_index + p] = wp.uint8(0)

                   # Update cell particle mapping buffer
                   ijk_float = wp.cw_div((particle_pos - particles.cell_particle_mapping.origin), particles.cell_particle_mapping.spacing)
                   particle_ijk = wp.vec3i(
                      wp.int32(ijk_float[0]),
                      wp.int32(ijk_float[1]),
                      wp.int32(ijk_float[2]),
                   )
                   wp.atomic_add(particles.cell_particle_mapping_buffer.data, particle_ijk[0], particle_ijk[1], particle_ijk[2], 1)

    def __call__(
        self,
        particles: Particles,
        mesh: wp.Mesh,
        nr_particles_per_cell: wp.int32,
        temperature: float,
        mean_velocity: wp.vec3f,
        seed: int = None,
    ):

        if seed is None:
            seed = np.random.randint(0, 2 ** 32)

        # Compute number of particles to add each cell
        wp.launch(
            self._add_particles,
            inputs=[
                particles,
                mesh.id,
                nr_particles_per_cell,
                wp.float32(temperature),
                mean_velocity,
                wp.int32(seed),
            ],
            dim=particles.cell_particle_mapping.shape,
        )

        # Update cell particle mapping
        wp.utils.array_scan(
            particles.cell_particle_mapping_buffer.data,
            particles.cell_particle_mapping.data,
            inclusive=False,
        )

        # Sort particles
        particles = self.part_sorter(particles)

        return particles
