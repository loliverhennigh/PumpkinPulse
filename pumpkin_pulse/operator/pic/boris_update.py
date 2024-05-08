import numpy as np
import warp as wp
from build123d import Compound
import tempfile
from stl import mesh as np_mesh

from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.operator.pic.interpolation import feild_to_particle_factory

class BorisUpdate(Operator):
    """
    Boris update operator
    """

    def __init__(
        shape_function: wp.func,
    ):
        # Set shape function
        self.shape_function = shape_function

        # Create feild to particle function
        self.feild_to_particle = feild_to_particle_factory(shape_function)

        # Initialize operator
        super().__init__()

    def make_kernels(self):

        # Function to interpolate electric field to particle
        @wp.func
        def _get_electric_field(
            pos: wp.vec3f,
            em_fields: EMFields,
        ):

            # Get origins for each component of the electric field
            origin_ex = (
                em_fields.origin - wp.vec3f(0.0, em_fields.spacing[1] / 2.0, spacing[2] / 2.0)
                - ghost_cell_offset
            )
            origin_ey = (
                origin
                - wp.vec3f(spacing[0] / 2.0, 0.0, spacing[2] / 2.0)
                - ghost_cell_offset
            )
            origin_ez = (
                origin
                - wp.vec3f(spacing[0] / 2.0, spacing[1] / 2.0, 0.0)
                - ghost_cell_offset
            )
    
            # Get lower cell index for E components
            f_ijk_ex = wp.cw_div(pos - origin_ex, spacing) - wp.vec3f(
                0.5, 0.5, 0.5
            )  # -0.5 to get the lower cell index
            f_ijk_ey = wp.cw_div(pos - origin_ey, spacing) - wp.vec3f(0.5, 0.5, 0.5)
            f_ijk_ez = wp.cw_div(pos - origin_ez, spacing) - wp.vec3f(0.5, 0.5, 0.5)
            ijk_ex = wp.vec3i(
                wp.int32(f_ijk_ex[0]), wp.int32(f_ijk_ex[1]), wp.int32(f_ijk_ex[2])
            )
            ijk_ey = wp.vec3i(
                wp.int32(f_ijk_ey[0]), wp.int32(f_ijk_ey[1]), wp.int32(f_ijk_ey[2])
            )
            ijk_ez = wp.vec3i(
                wp.int32(f_ijk_ez[0]), wp.int32(f_ijk_ez[1]), wp.int32(f_ijk_ez[2])
            )
    
            # Get electric field
            e_x = BorisVelocityUpdate._feild_to_particle(
                pos, ijk_ex, electric_field, 0, origin_ex, spacing
            )
            e_y = BorisVelocityUpdate._feild_to_particle(
                pos, ijk_ey, electric_field, 1, origin_ey, spacing
            )
            e_z = BorisVelocityUpdate._feild_to_particle(
                pos, ijk_ez, electric_field, 2, origin_ez, spacing
            )
    
            return wp.vec3f(e_x, e_y, e_z)


        @wp.kernel
        def _update_velocity(
            particle_position: wp.array2d(dtype=wp.float32),
            particle_velocity: wp.array2d(dtype=wp.float32),
            particle_id: wp.array2d(dtype=wp.uint8),
            electric_field: wp.array4d(dtype=wp.float32),
            magnetic_field: wp.array4d(dtype=wp.float32),
            particle_mass_mapping: wp.array(dtype=wp.float32),
            particle_charge_mapping: wp.array(dtype=wp.float32),
            origin: wp.vec3f,
            spacing: wp.vec3f,
            dt: wp.float32,
            nr_ghost_cells: wp.int32,
        ):
            # get particle index
            i = wp.tid()
    
            # Get particle id
            pid = particle_id[0, i]
    
            # If particle is not active, return
            if pid == 0:  # zero is the id for dead particles
                return
    
            # Get particle position and velocity
            pos = wp.vec3f(
                particle_position[0, i], particle_position[1, i], particle_position[2, i]
            )
            vel = wp.vec3f(
                particle_velocity[0, i], particle_velocity[1, i], particle_velocity[2, i]
            )
    
            # Get properties
            mass = particle_mass_mapping[wp.int32(pid)]
            charge = particle_charge_mapping[wp.int32(pid)]
            charge_mass_ratio = charge / mass
    
            # Get electric and magnetic field
            e = BorisVelocityUpdate._get_electric_field(
                pos, electric_field, origin, spacing, nr_ghost_cells
            )
            b = BorisVelocityUpdate._get_magnetic_field(
                pos, magnetic_field, origin, spacing, nr_ghost_cells
            )
    
            # First half step of electric field
            vel_minus = vel + 0.5 * charge_mass_ratio * dt * e
    
            # Rotation due to magnetic field
            t = b * (0.5 * charge_mass_ratio * dt)
            s = 2.0 * t / (1.0 + wp.dot(t, t))
            vel_prime = vel_minus + wp.cross(vel_minus, t)
            vel_plus = vel_minus + wp.cross(vel_prime, s)
    
            # Second half step of electric field
            new_vel = vel_plus + 0.5 * charge_mass_ratio * dt * e
    
            # Set new velocity
            particle_velocity[0, i] = new_vel[0]
            particle_velocity[1, i] = new_vel[1]
            particle_velocity[2, i] = new_vel[2]


    @wp.func
    def _get_electric_field(
        pos: wp.vec3f,
        electric_field: wp.array4d(dtype=wp.float32),
        origin: wp.vec3f,
        spacing: wp.vec3f,
        nr_ghost_cells: wp.int32,
    ):
        # Get ghost cell offset
        ghost_cell_offset = wp.vec3f(
            spacing[0] * wp.float32(nr_ghost_cells),
            spacing[1] * wp.float32(nr_ghost_cells),
            spacing[2] * wp.float32(nr_ghost_cells),
        )

        # Get origins for each component of the electric field
        origin_ex = (
            origin
            - wp.vec3f(0.0, spacing[1] / 2.0, spacing[2] / 2.0)
            - ghost_cell_offset
        )
        origin_ey = (
            origin
            - wp.vec3f(spacing[0] / 2.0, 0.0, spacing[2] / 2.0)
            - ghost_cell_offset
        )
        origin_ez = (
            origin
            - wp.vec3f(spacing[0] / 2.0, spacing[1] / 2.0, 0.0)
            - ghost_cell_offset
        )

        # Get lower cell index for E components
        f_ijk_ex = wp.cw_div(pos - origin_ex, spacing) - wp.vec3f(
            0.5, 0.5, 0.5
        )  # -0.5 to get the lower cell index
        f_ijk_ey = wp.cw_div(pos - origin_ey, spacing) - wp.vec3f(0.5, 0.5, 0.5)
        f_ijk_ez = wp.cw_div(pos - origin_ez, spacing) - wp.vec3f(0.5, 0.5, 0.5)
        ijk_ex = wp.vec3i(
            wp.int32(f_ijk_ex[0]), wp.int32(f_ijk_ex[1]), wp.int32(f_ijk_ex[2])
        )
        ijk_ey = wp.vec3i(
            wp.int32(f_ijk_ey[0]), wp.int32(f_ijk_ey[1]), wp.int32(f_ijk_ey[2])
        )
        ijk_ez = wp.vec3i(
            wp.int32(f_ijk_ez[0]), wp.int32(f_ijk_ez[1]), wp.int32(f_ijk_ez[2])
        )

        # Get electric field
        e_x = BorisVelocityUpdate._feild_to_particle(
            pos, ijk_ex, electric_field, 0, origin_ex, spacing
        )
        e_y = BorisVelocityUpdate._feild_to_particle(
            pos, ijk_ey, electric_field, 1, origin_ey, spacing
        )
        e_z = BorisVelocityUpdate._feild_to_particle(
            pos, ijk_ez, electric_field, 2, origin_ez, spacing
        )

        return wp.vec3f(e_x, e_y, e_z)

    @wp.func
    def _get_magnetic_field(
        pos: wp.vec3f,
        magnetic_field: wp.array4d(dtype=wp.float32),
        origin: wp.vec3f,
        spacing: wp.vec3f,
        nr_ghost_cells: wp.int32,
    ):
        # Get ghost cell offset
        ghost_cell_offset = wp.vec3f(
            spacing[0] * wp.float32(nr_ghost_cells),
            spacing[1] * wp.float32(nr_ghost_cells),
            spacing[2] * wp.float32(nr_ghost_cells),
        )

        # Get origins for each component of the magnetic field
        origin_bx = origin - wp.vec3f(spacing[0] / 2.0, 0.0, 0.0) - ghost_cell_offset
        origin_by = origin - wp.vec3f(0.0, spacing[1] / 2.0, 0.0) - ghost_cell_offset
        origin_bz = origin - wp.vec3f(0.0, 0.0, spacing[2] / 2.0) - ghost_cell_offset

        # Get cell index for B
        f_ijk_bx = wp.cw_div(pos - origin_bx, spacing) - wp.vec3f(
            0.5, 0.5, 0.5
        )  # -0.5 to get the lower cell index
        f_ijk_by = wp.cw_div(pos - origin_by, spacing) - wp.vec3f(0.5, 0.5, 0.5)
        f_ijk_bz = wp.cw_div(pos - origin_bz, spacing) - wp.vec3f(0.5, 0.5, 0.5)
        ijk_bx = wp.vec3i(
            wp.int32(f_ijk_bx[0]), wp.int32(f_ijk_bx[1]), wp.int32(f_ijk_bx[2])
        )
        ijk_by = wp.vec3i(
            wp.int32(f_ijk_by[0]), wp.int32(f_ijk_by[1]), wp.int32(f_ijk_by[2])
        )
        ijk_bz = wp.vec3i(
            wp.int32(f_ijk_bz[0]), wp.int32(f_ijk_bz[1]), wp.int32(f_ijk_bz[2])
        )

        # Get magnetic field
        b_x = BorisVelocityUpdate._feild_to_particle(
            pos, ijk_bx, magnetic_field, 0, origin_bx, spacing
        )
        b_y = BorisVelocityUpdate._feild_to_particle(
            pos, ijk_by, magnetic_field, 1, origin_by, spacing
        )
        b_z = BorisVelocityUpdate._feild_to_particle(
            pos, ijk_bz, magnetic_field, 2, origin_bz, spacing
        )

        return wp.vec3f(b_x, b_y, b_z)
    def __call__(
        self,
        particles: Particles,
        em_fields: EMFields,
        dt: float,
    ):
        # Launch kernel
        wp.launch(
            self._boris_push,
            inputs=[
                particles,
                em_fields
                dt,
                nr_ghost_cells,
            ],
            dim=particles.num_particles,
        )

        return particle_velocity

