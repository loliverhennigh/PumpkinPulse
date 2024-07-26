import warp as wp

from pumpkin_pulse.struct.field import Fieldfloat32
from pumpkin_pulse.struct.particles import Particles
from pumpkin_pulse.struct.material_properties import MaterialProperties
from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.functional.indexing import pos_to_cell_index, pos_to_lower_cell_index
from pumpkin_pulse.functional.marching_cube import VERTEX_TABLE, VERTEX_INDICES_TABLE
from pumpkin_pulse.functional.ray_triangle_intersect import ray_triangle_intersect


class HardSphereCollision(Operator):
    """
    Hard sphere collision operator
    """

    @wp.func
    def get_neighbour_cell_indices(
        cell_particle_mapping: wp.array3d(dtype=wp.int32),
        i: wp.int32,
        j: wp.int32,
        k: wp.int32,
    ) -> wp.int32:

        # Convert 3d index to 1d index
        index = (i * cell_particle_mapping.shape[1] + j) * cell_particle_mapping.shape[2] + k

        # Add 1 to index
        index += 1

        # Convert to 3d index again
        i = index // (cell_particle_mapping.shape[1] * cell_particle_mapping.shape[2])
        j = (index % (cell_particle_mapping.shape[1] * cell_particle_mapping.shape[2])) // cell_particle_mapping.shape[2]
        k = index % cell_particle_mapping.shape[2]

        return cell_particle_mapping[i, j, k]

    # Make push kernel
    @wp.kernel
    def collision(
        particles: Particles,
        dt: wp.float32,
    ):

        # Get cell index
        i, j, k = wp.tid()

        # Break if last cell
        if i == particles.cell_particle_mapping.shape[0] - 1:
            if j == particles.cell_particle_mapping.shape[1] - 1:
                if k == particles.cell_particle_mapping.shape[2] - 1:
                    return

        # Get index in particle array
        p_index = particles.cell_particle_mapping.data[i, j, k]

        # Get number of particles in cell
        n_particles = HardSphereCollision.get_neighbour_cell_indices(
            particles.cell_particle_mapping.data, i, j, k
        ) - p_index

        # Collide particles in cell
        v_rel_max = 2.0 * particles.cell_particle_mapping.spacing[0] / dt

        # Compute number of candidate pairs
        m_candidates = wp.int32(
            wp.float32(n_particles)**2.0
            * 3.14159
            * v_rel_max
            * particles.weighting
            * dt
            / (2.0 * particles.cell_particle_mapping.spacing[0]**3.0)
        ) / 200000
        #if m_candidates > 0:
        #    print(m_candidates)

        # Initialize random number generator
        r = wp.rand_init(123, i * particles.cell_particle_mapping.shape[1] * particles.cell_particle_mapping.shape[2] + j * particles.cell_particle_mapping.shape[2] + k)

        # Loop over candidate pairs
        for m in range(m_candidates):

            # Get random particle indices
            p_index_0 = wp.randi(r, 0, n_particles)
            p_index_1 = wp.randi(r, 0, n_particles)

            # Get particle momentums
            mom_0 = particles.momentum[p_index_0 + p_index]
            mom_1 = particles.momentum[p_index_1 + p_index]

            # Get velocities
            vel_0 = mom_0 / particles.mass
            vel_1 = mom_1 / particles.mass

            # Get relative velocity
            v_rel = wp.length(vel_0 - vel_1)

            # Get random number
            r_frac = wp.randf(r)

            # Check if collision occurs
            if v_rel > v_rel_max * r_frac:

                # Hard sphere collision
                v_cm = 0.5 * (vel_0 + vel_1)
                cos_theta = 2.0 * wp.randf(r) - 1.0
                sin_theta = wp.sqrt(1.0 - cos_theta**2.0)
                phi = 2.0 * 3.14159 * wp.randf(r)
                v_p = wp.vec3(
                    v_rel * sin_theta * wp.cos(phi),
                    v_rel * sin_theta * wp.sin(phi),
                    v_rel * cos_theta,
                )

                # Update velocities
                vel_0 = v_cm + 0.5 * v_p
                vel_1 = v_cm - 0.5 * v_p

                # Update momentums
                mom_0 = particles.mass * vel_0
                mom_1 = particles.mass * vel_1

                # Update particle momentum
                particles.momentum[p_index_0 + p_index] = mom_0
                particles.momentum[p_index_1 + p_index] = mom_1

    def __call__(
        self,
        particles: Particles,
        dt: float,
    ):

        # Compute number of particles to add each cel
        wp.launch(
            self.collision,
            inputs=[
                particles,
                dt,
            ],
            dim=(particles.cell_particle_mapping.shape[0], particles.cell_particle_mapping.shape[1], particles.cell_particle_mapping.shape[2]),
        )

        return particles
