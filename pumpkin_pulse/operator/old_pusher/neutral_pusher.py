import warp as wp

from pumpkin_pulse.struct.particles import Particles, Particle
from pumpkin_pulse.struct.material_properties import MaterialProperties
from pumpkin_pulse.operator.pusher.pusher import Pusher


class NeutralPusher(Pusher):
    """
    Pushes particles that have no charge in time
    """

    def __init__(
        self,
        boundary_conditions: str = "periodic",
    ):
        super().__init__(boundary_conditions)

        # Make push kernel
        @wp.kernel
        def push(
            particles: Particles,
            material_properties: MaterialProperties,
            dt: wp.float32,
        ):

            # Get particle index
            i = wp.tid()

            # Get particle data
            d = particles.data[i]

            # Get particles momentum and position
            mom = d.momentum
            pos = d.position

            # Store remaining time to push particle
            remaining_dt = dt

            # Allocate pushed particle
            pushed_particle = Particle(
                pos,
                mom,
                wp.uint8(0), # Keep particle
            )

            # Get epsilon distance
            epsilon = particles.cell_particle_mapping.spacing[0] * 1e-3

            # Move particle through cell walls and check for solid interactions
            for _ in range(100):

                # Get sdf for current position
                sdf = Pusher.solid_sdf(
                    pos,
                    material_properties,
                    particles,
                )

                # Get velocity
                v = mom / particles.mass

                # Compute maximum push time
                push_dt = wp.min((sdf - epsilon) / wp.length(v), remaining_dt)

                # Push particle
                new_pos = pos + push_dt * v

                # Get sdf for new position
                new_sdf = Pusher.solid_sdf(
                    new_pos,
                    material_properties,
                    particles,
                )

                # Check if particle is in solid
                if new_sdf < (2.0 * epsilon):

                    #if new_sdf < 0.0:
                    #    pushed_particle.position = pos
                    #    print(new_sdf)
                    #    break

                    # Linear interpolate to get new position at 2*epsilon
                    push_dt = push_dt * (sdf - 2.0 * epsilon) / (sdf - new_sdf)

                    # Reflective
                    new_pos = pos + push_dt * v

                    mom = -mom
                    epsilon = -epsilon

                # Update remaining time
                remaining_dt -= push_dt

                # Update position
                pos = new_pos

                # Check if particle needs to be pushed
                if remaining_dt <= 0.0:
                    pushed_particle.position = pos
                    pushed_particle.momentum = mom
                    pushed_particle.kill = wp.uint8(0)
                    break

            # Set new position
            particles.data[i] = pushed_particle

            # Get index of new cell
            ijk = Pusher.pos_to_cell_index(
                pushed_particle.position,
                particles.cell_particle_mapping.origin,
                particles.cell_particle_mapping.spacing,
                particles.cell_particle_mapping.nr_ghost_cells
            )

            # Add particle count to particles per cell
            wp.atomic_add(particles.cell_particle_mapping_buffer.data, ijk[0], ijk[1], ijk[2], 1)

        # Store push kernel
        self.push = push

    def __call__(
        self,
        particles: Particles,
        material_properties: MaterialProperties,
        dt: float,
    ):

        # Zero cell particle mapping buffer
        particles.cell_particle_mapping_buffer.data.zero_()

        # Compute number of particles to add each cel
        wp.launch(
            self.push,
            inputs=[
                particles,
                material_properties,
                dt,
            ],
            dim=(particles.nr_particles.numpy()[0],),
        )

        # Update cell particle mapping
        wp.utils.array_scan(
            particles.cell_particle_mapping_buffer.data,
            particles.cell_particle_mapping.data,
            inclusive=False,
        )

        # Sort particles
        particles = Pusher.sort_particles(particles)

        return particles
