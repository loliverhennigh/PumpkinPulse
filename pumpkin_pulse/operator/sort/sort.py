import warp as wp

from pumpkin_pulse.struct.particles import Particles
from pumpkin_pulse.operator.operator import Operator

class ParticleSorter(Operator):

    @wp.kernel
    def _sort_particles(
        particles: Particles,
    ):
        # Get particle index
        i = wp.tid()

        # Load particle data
        d = particles.data[i]

        # Determine cell index
        ijk_f = wp.cw_div((d.position - particles.rho_origin), particles.spacing)
        ijk = wp.vec3i(
            wp.int32(wp.floor(ijk_f[0])) + particles.nr_ghost_cells,
            wp.int32(wp.floor(ijk_f[1])) + particles.nr_ghost_cells,
            wp.int32(wp.floor(ijk_f[2])) + particles.nr_ghost_cells,
        )

        # Get cell particle mapping index
        particle_index = particles.cell_particle_mapping[ijk[0], ijk[1], ijk[2]]

        # Get offset with buffer
        offset = wp.atomic_add(particles.cell_particle_mapping_buffer, ijk[0], ijk[1], ijk[2], 1)

        # Add particle to cell
        particles.data_buffer[offset + particle_index] = d

    def __call__(
        self,
        particles: Particles,
    ):

        # Zero cell particle mapping buffer
        particles.cell_particle_mapping_buffer.zero_()

        # Compute number of particles to add each cell
        wp.launch(
            self._sort_particles,
            inputs=[
                particles,
            ],
            dim=(particles.nr_particles.numpy()[0],),
        )

        # Rotate buffers
        particles.data, particles.data_buffer = particles.data_buffer, particles.data

        return particles
