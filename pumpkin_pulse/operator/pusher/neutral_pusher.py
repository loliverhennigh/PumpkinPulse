import warp as wp

from pumpkin_pulse.struct.particles import Particles, PosMom
from pumpkin_pulse.struct.material_properties import MaterialProperties
from pumpkin_pulse.operator.pusher.pusher import Pusher

@wp.func
def trilinear_interpolation(
    material_properties: MaterialProperties,
    pos: wp.vec3,
):
    # Get lower cell index
    float_ijk = (wp.cw_div(pos - material_properties.origin, material_properties.spacing)
        + wp.vec3(
            wp.float32(material_properties.nr_ghost_cells), 
            wp.float32(material_properties.nr_ghost_cells),
            wp.float32(material_properties.nr_ghost_cells)
        )
    )
    ijk = wp.vec3i(wp.int32(float_ijk[0]), wp.int32(float_ijk[1]), wp.int32(float_ijk[2]))

    # Get id for all corners of cell
    f_000 = material_properties.solid_mapping[wp.int32(material_properties.id[ijk[0], ijk[1], ijk[2]])]
    f_100 = material_properties.solid_mapping[wp.int32(material_properties.id[ijk[0] + 1, ijk[1], ijk[2]])]
    f_010 = material_properties.solid_mapping[wp.int32(material_properties.id[ijk[0], ijk[1] + 1, ijk[2]])]
    f_110 = material_properties.solid_mapping[wp.int32(material_properties.id[ijk[0] + 1, ijk[1] + 1, ijk[2]])]
    f_001 = material_properties.solid_mapping[wp.int32(material_properties.id[ijk[0], ijk[1], ijk[2] + 1])]
    f_101 = material_properties.solid_mapping[wp.int32(material_properties.id[ijk[0] + 1, ijk[1], ijk[2] + 1])]
    f_011 = material_properties.solid_mapping[wp.int32(material_properties.id[ijk[0], ijk[1] + 1, ijk[2] + 1])]
    f_111 = material_properties.solid_mapping[wp.int32(material_properties.id[ijk[0] + 1, ijk[1] + 1, ijk[2] + 1])]

    # Get relative position in cell
    relative_pos = float_ijk - wp.vec3(wp.float32(ijk[0]), wp.float32(ijk[1]), wp.float32(ijk[2]))

    # x-direction
    f_00 = f_000 * (1.0 - relative_pos[0]) + f_100 * relative_pos[0]
    f_01 = f_001 * (1.0 - relative_pos[0]) + f_101 * relative_pos[0]
    f_10 = f_010 * (1.0 - relative_pos[0]) + f_110 * relative_pos[0]
    f_11 = f_011 * (1.0 - relative_pos[0]) + f_111 * relative_pos[0]

    # y-direction
    f_0 = f_00 * (1.0 - relative_pos[1]) + f_10 * relative_pos[1]
    f_1 = f_01 * (1.0 - relative_pos[1]) + f_11 * relative_pos[1]

    # z-direction
    f = f_0 * (1.0 - relative_pos[2]) + f_1 * relative_pos[2]

    return f

@wp.func
def pos_to_cell(
    pos: wp.vec3f,
    origin: wp.vec3f,
    spacing: wp.vec3f,
    nr_ghost_cells: wp.int32,
):
    return wp.vec3i(
        wp.int32((pos[0] - origin[0]) / spacing[0]) + nr_ghost_cells,
        wp.int32((pos[1] - origin[1]) / spacing[1]) + nr_ghost_cells,
        wp.int32((pos[2] - origin[2]) / spacing[2]) + nr_ghost_cells,
    )

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

            # Get particle momentum
            mom = d.momentum

            # Compute new position (Bounce of any walls from material properties)
            old_pos = d.position
            new_pos = wp.vec3()

            # Get solid fraction at old position
            old_f = trilinear_interpolation(material_properties, old_pos)

            # Check for reflections (Bounce of any walls from material properties)
            for _ in range(10): # 10 as maximum number of reflections

                # Compute new position
                new_pos = old_pos + (dt * mom / particles.mass)

                # Get solid fraction at new position
                new_f = trilinear_interpolation(material_properties, new_pos)

                # Check if particle is in solid
                if new_f > 0.5:

                    # Get wall kind
                    wall_kind = material_properties.kind_mapping[wp.int32(material_properties.id[0, 0, 0])]

                    # Find time of intersection
                    dt = dt * (0.5 - old_f) / (new_f - old_f)

                    # Get new position and use it as old position
                    old_pos = old_pos + (dt * mom / particles.mass)

                    # Reflect momentum
                    mom = -mom

                else:
                    break


            # Apply boundary conditions
            new_pos = self.apply_boundary_conditions(
                new_pos,
                particles.origin,
                particles.spacing,
                particles.shape,
                particles.nr_ghost_cells
            )

            # Set new position
            pos_mom = PosMom(new_pos, mom)
            particles.data[i] = pos_mom

            # Get index of new cell
            ijk = pos_to_cell(
                new_pos,
                particles.rho_origin,
                particles.spacing,
                particles.nr_ghost_cells
            )

            # Add particle count to particles per cell
            wp.atomic_add(particles.cell_particle_mapping_buffer, ijk[0], ijk[1], ijk[2], 1)

        # Store push kernel
        self.push = push

    def __call__(
        self,
        particles: Particles,
        material_properties: MaterialProperties,
        dt: float,
    ):

        # Zero cell particle mapping buffer
        particles.cell_particle_mapping_buffer.zero_()

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
            particles.cell_particle_mapping_buffer,
            particles.cell_particle_mapping,
            inclusive=False,
        )

        # Sort particles
        particles = Pusher.sort_particles(particles)

        return particles
