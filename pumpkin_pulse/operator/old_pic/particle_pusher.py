import numpy as np
import warp as wp
from build123d import Compound
import tempfile
from stl import mesh as np_mesh

from pumpkin_pulse.operator.operator import Operator

@wp.func
def linear_shape_function(
    x: wp.float32,
    x_ijk: wp.float32,
    spacing: wp.float32,
):
    w = 1.0 - wp.abs((x - x_ijk) / spacing)
    w = wp.max(w, 0.0)
    return w

class BorisVelocityUpdate(Operator):
    """
    Boris velocity update operator
    """

    @wp.func
    def _feild_to_particle(
        pos: wp.vec3f,
        ijk_lower: wp.vec3i,
        feild: wp.array4d(dtype=wp.float32),
        feild_index: wp.int32,
        feild_origin: wp.vec3f,
        feild_spacing: wp.vec3f,
    ) -> wp.float32:
        # Get feild values at corners
        f_0_0_0 = feild[feild_index, ijk_lower[0], ijk_lower[1], ijk_lower[2]]
        f_1_0_0 = feild[feild_index, ijk_lower[0] + 1, ijk_lower[1], ijk_lower[2]]
        f_0_1_0 = feild[feild_index, ijk_lower[0], ijk_lower[1] + 1, ijk_lower[2]]
        f_1_1_0 = feild[feild_index, ijk_lower[0] + 1, ijk_lower[1] + 1, ijk_lower[2]]
        f_0_0_1 = feild[feild_index, ijk_lower[0], ijk_lower[1], ijk_lower[2] + 1]
        f_1_0_1 = feild[feild_index, ijk_lower[0] + 1, ijk_lower[1], ijk_lower[2] + 1]
        f_0_1_1 = feild[feild_index, ijk_lower[0], ijk_lower[1] + 1, ijk_lower[2] + 1]
        f_1_1_1 = feild[
            feild_index, ijk_lower[0] + 1, ijk_lower[1] + 1, ijk_lower[2] + 1
        ]

        # Get position for each dimension
        cell_pos_x_lower = (
            feild_origin[0]
            + wp.float32(ijk_lower[0]) * feild_spacing[0]
            + feild_spacing[0] / 2.0
        )
        cell_pos_y_lower = (
            feild_origin[1]
            + wp.float32(ijk_lower[1]) * feild_spacing[1]
            + feild_spacing[1] / 2.0
        )
        cell_pos_z_lower = (
            feild_origin[2]
            + wp.float32(ijk_lower[2]) * feild_spacing[2]
            + feild_spacing[2] / 2.0
        )
        cell_pos_x_upper = cell_pos_x_lower + feild_spacing[0]
        cell_pos_y_upper = cell_pos_y_lower + feild_spacing[1]
        cell_pos_z_upper = cell_pos_z_lower + feild_spacing[2]

        # Get shape functions for each corner
        s_x_lower = linear_shape_function(pos[0], cell_pos_x_lower, feild_spacing[0])
        s_y_lower = linear_shape_function(pos[1], cell_pos_y_lower, feild_spacing[1])
        s_z_lower = linear_shape_function(pos[2], cell_pos_z_lower, feild_spacing[2])
        s_x_upper = linear_shape_function(pos[0], cell_pos_x_upper, feild_spacing[0])
        s_y_upper = linear_shape_function(pos[1], cell_pos_y_upper, feild_spacing[1])
        s_z_upper = linear_shape_function(pos[2], cell_pos_z_upper, feild_spacing[2])

        # Interpolate feild
        f = (
            f_0_0_0 * s_x_lower * s_y_lower * s_z_lower
            + f_1_0_0 * s_x_upper * s_y_lower * s_z_lower
            + f_0_1_0 * s_x_lower * s_y_upper * s_z_lower
            + f_1_1_0 * s_x_upper * s_y_upper * s_z_lower
            + f_0_0_1 * s_x_lower * s_y_lower * s_z_upper
            + f_1_0_1 * s_x_upper * s_y_lower * s_z_upper
            + f_0_1_1 * s_x_lower * s_y_upper * s_z_upper
            + f_1_1_1 * s_x_upper * s_y_upper * s_z_upper
        )

        return f

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

    def __call__(
        self,
        particle_position: wp.array2d(dtype=wp.float32),
        particle_velocity: wp.array2d(dtype=wp.float32),
        particle_id: wp.array2d(dtype=wp.uint8),
        electric_field: wp.array4d(dtype=wp.float32),
        magnetic_field: wp.array4d(dtype=wp.float32),
        particle_mass_mapping: wp.array(dtype=wp.float32),
        particle_charge_mapping: wp.array(dtype=wp.float32),
        origin: tuple[float, float, float],
        spacing: tuple[float, float, float],
        dt: float,
        nr_ghost_cells: int = 1,
    ):
        # Launch kernel
        wp.launch(
            self._update_velocity,
            inputs=[
                particle_position,
                particle_velocity,
                particle_id,
                electric_field,
                magnetic_field,
                particle_mass_mapping,
                particle_charge_mapping,
                origin,
                spacing,
                dt,
                nr_ghost_cells,
            ],
            dim=particle_position.shape[1],
        )

        return particle_velocity


class PositionUpdate(Operator):
    """
    Position update operator
    """

    s_mat = wp.mat((5, 3), wp.float32)
    current_mat = wp.mat((125, 3), wp.float32)

    @wp.func
    def _move_particle(
        particle_index: wp.int32,
        pos: wp.vec3f,
        vel: wp.vec3f,
        particle_id: wp.array2d(dtype=wp.uint8),
        material_id: wp.array4d(dtype=wp.uint8),
        permiable_mapping: wp.array(dtype=wp.uint8),
        origin: wp.vec3f,
        spacing: wp.vec3f,
        dt: wp.float32,
        nr_ghost_cells: wp.int32,
    ) -> wp.vec3f:
        # Get ghost cell offset
        ghost_cell_offset = wp.vec3f(
            spacing[0] * wp.float32(nr_ghost_cells),
            spacing[1] * wp.float32(nr_ghost_cells),
            spacing[2] * wp.float32(nr_ghost_cells),
        )

        # Get origin for material id
        origin_material = origin - ghost_cell_offset

        # Get cell index for material id
        f_ijk_material = wp.cw_div(pos - origin_material, spacing)
        ijk_material = wp.vec3i(
            wp.int32(f_ijk_material[0]),
            wp.int32(f_ijk_material[1]),
            wp.int32(f_ijk_material[2]),
        )

        # See if particle hits a boundary
        min_dt = dt
        # for d in range(3):
        #    # Get time to face
        #    if (vel[d] > 0.0) and (material_id[wp.int32(ijk_material[0]), wp.int32(ijk_material[1]), wp.int32(ijk_material[2])] != 0):
        #        time_to_face = (wp.float32(ijk_material[d] + 1) - f_ijk_material[d]) * spacing[d] / vel[d]
        #    elif vel[d] < 0.0:
        #        time_to_face = (f_ijk_material[d] - wp.float32(ijk_material[d])) * spacing[d] / vel[d]
        #    else:
        #        time_to_face = dt

        #    # Check if time to face is smaller than min_dt, if so, update min_dt and kill particle
        #    if time_to_face < min_dt:
        #        min_dt = time_to_face
        #        #particle_id[0, particle_index] = wp.uint8(0) # Kill particle

        # Get material id
        m_id = material_id[
            0,
            wp.int32(ijk_material[0]),
            wp.int32(ijk_material[1]),
            wp.int32(ijk_material[2]),
        ]

        # Check if particle in a permiable cell
        if permiable_mapping[wp.int32(m_id)] != wp.uint8(1):
            particle_id[0, particle_index] = wp.uint8(0)  # Kill particle

        # Move particle
        new_pos = pos + vel * min_dt

        return new_pos

    @wp.func
    def _w_helper_function(
        s_1: wp.float32,
        s_2: wp.float32,
        s_3: wp.float32,
        s_4: wp.float32,
        s_5: wp.float32,
        s_6: wp.float32,
    ):
        s_456 = s_4 * s_5 * s_6
        s_156 = s_1 * s_5 * s_6
        s_423 = s_4 * s_2 * s_3
        s_123 = s_1 * s_2 * s_3
        term_1 = (1.0 / 3.0) * (s_456 - s_156 + s_423 - s_123)
        s_426 = s_4 * s_2 * s_6
        s_126 = s_1 * s_2 * s_6
        s_453 = s_4 * s_5 * s_3
        s_153 = s_1 * s_5 * s_3
        term_2 = (1.0 / 6.0) * (s_426 - s_126 + s_453 - s_153)
        return term_1 + term_2

    @wp.func
    def _current_deposition(
        old_pos: wp.vec3f,
        new_pos: wp.vec3f,
        mass: wp.float32,
        charge: wp.float32,
        current_density: wp.array4d(dtype=wp.float32),
        origin: wp.vec3f,
        spacing: wp.vec3f,
        dt: wp.float32,
        nr_ghost_cells: wp.int32,
    ):
        # Get ghost cell offset
        ghost_cell_offset = wp.vec3f(
            spacing[0] * wp.float32(nr_ghost_cells),
            spacing[1] * wp.float32(nr_ghost_cells),
            spacing[2] * wp.float32(nr_ghost_cells),
        )

        # Get cell index for charge density
        current_density_origin = (
            origin
            - wp.vec3f(spacing[0] / 2.0, spacing[1] / 2.0, spacing[2] / 2.0)
            - ghost_cell_offset
        )

        # Get nearest cell index for charge density
        f_ijk_rho = wp.cw_div(old_pos - current_density_origin, spacing)
        ijk_rho = wp.vec3i(
            wp.int32(f_ijk_rho[0]), wp.int32(f_ijk_rho[1]), wp.int32(f_ijk_rho[2])
        )

        # Find all shape functions
        s_old = PositionUpdate.s_mat()
        s_new = PositionUpdate.s_mat()
        for i in range(5):
            for d in range(3):
                # Get cell location for shape function
                cell_pos = (
                    wp.float32(ijk_rho[d] + i - 2) * spacing[d]
                    + current_density_origin[d]
                    + spacing[d] / 2.0  # Center of the cell
                )

                # Calculate shape functions
                s_old[i, d] = linear_shape_function(old_pos[d], cell_pos, spacing[d])
                s_new[i, d] = linear_shape_function(new_pos[d], cell_pos, spacing[d])

        # Calculate current deposition vector
        current_mat = PositionUpdate.current_mat()
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    # Calculate current deposition vector
                    w_x = PositionUpdate._w_helper_function(
                        s_old[i, 0],
                        s_old[j, 1],
                        s_old[k, 2],
                        s_new[i, 0],
                        s_new[j, 1],
                        s_new[k, 2],
                    )
                    w_y = PositionUpdate._w_helper_function(
                        s_old[j, 1],
                        s_old[i, 0],
                        s_old[k, 2],
                        s_new[j, 1],
                        s_new[i, 0],
                        s_new[k, 2],
                    )
                    w_z = PositionUpdate._w_helper_function(
                        s_old[k, 2],
                        s_old[j, 1],
                        s_old[i, 0],
                        s_new[k, 2],
                        s_new[j, 1],
                        s_new[i, 0],
                    )

                    # Calculate current density contribution
                    diff_J_x = -charge * (spacing[0] / dt) * w_x
                    diff_J_y = -charge * (spacing[1] / dt) * w_y
                    diff_J_z = -charge * (spacing[2] / dt) * w_z

                    # Deposit current density
                    current_mat[i * 25 + j * 5 + k, 0] = diff_J_x
                    current_mat[i * 25 + j * 5 + k, 1] = diff_J_y
                    current_mat[i * 25 + j * 5 + k, 2] = diff_J_z
                    if i != 0:
                        current_mat[i * 25 + j * 5 + k, 0] += current_mat[
                            (i - 1) * 25 + j * 5 + k, 0
                        ]
                    if j != 0:
                        current_mat[i * 25 + j * 5 + k, 1] += current_mat[
                            i * 25 + (j - 1) * 5 + k, 1
                        ]
                    if k != 0:
                        current_mat[i * 25 + j * 5 + k, 2] += current_mat[
                            i * 25 + j * 5 + (k - 1), 2
                        ]

        # Add current matrix to global current density
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    wp.atomic_add(
                        current_density,
                        0,
                        wp.int32(ijk_rho[0]) + i - 2,
                        wp.int32(ijk_rho[1]) + j - 2,
                        wp.int32(ijk_rho[2]) + k - 2,
                        current_mat[i * 25 + j * 5 + k, 0],
                    )
                    wp.atomic_add(
                        current_density,
                        1,
                        wp.int32(ijk_rho[0]) + i - 2,
                        wp.int32(ijk_rho[1]) + j - 2,
                        wp.int32(ijk_rho[2]) + k - 2,
                        current_mat[i * 25 + j * 5 + k, 1],
                    )
                    wp.atomic_add(
                        current_density,
                        2,
                        wp.int32(ijk_rho[0]) + i - 2,
                        wp.int32(ijk_rho[1]) + j - 2,
                        wp.int32(ijk_rho[2]) + k - 2,
                        current_mat[i * 25 + j * 5 + k, 2],
                    )

    @wp.kernel
    def _update_position(
        particle_position: wp.array2d(dtype=wp.float32),
        particle_velocity: wp.array2d(dtype=wp.float32),
        particle_id: wp.array2d(dtype=wp.uint8),
        current_density: wp.array4d(dtype=wp.float32),
        particle_mass_mapping: wp.array(dtype=wp.float32),
        particle_charge_mapping: wp.array(dtype=wp.float32),
        material_id: wp.array4d(dtype=wp.uint8),
        permiable_mapping: wp.array(dtype=wp.uint8),
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

        # Get new position
        new_pos = PositionUpdate._move_particle(
            i, pos, vel, particle_id, material_id, permiable_mapping,
            origin, spacing, dt, nr_ghost_cells
        )

        # If particle is out of bounds, kill it
        if new_pos[0] < origin[0] - 2.0 * spacing[1] or new_pos[0] > origin[
            0
        ] + spacing[0] * wp.float32(current_density.shape[1] - 4 * nr_ghost_cells):
            particle_id[0, i] = wp.uint8(0)
            return
        if new_pos[1] < origin[1] - 2.0 * spacing[1] or new_pos[1] > origin[
            1
        ] + spacing[1] * wp.float32(current_density.shape[2] - 4 * nr_ghost_cells):
            particle_id[0, i] = wp.uint8(0)
            return
        if new_pos[2] < origin[2] - 2.0 * spacing[1] or new_pos[2] > origin[
            2
        ] + spacing[2] * wp.float32(current_density.shape[3] - 4 * nr_ghost_cells):
            particle_id[0, i] = wp.uint8(0)
            return

        # Deposit current density
        PositionUpdate._current_deposition(
            pos,
            new_pos,
            mass,
            charge,
            current_density,
            origin,
            spacing,
            dt,
            nr_ghost_cells,
        )

        # Set new position
        particle_position[0, i] = new_pos[0]
        particle_position[1, i] = new_pos[1]
        particle_position[2, i] = new_pos[2]

    def __call__(
        self,
        particle_position: wp.array2d(dtype=wp.float32),
        particle_velocity: wp.array2d(dtype=wp.float32),
        particle_id: wp.array2d(dtype=wp.uint8),
        current_density: wp.array4d(dtype=wp.float32),
        particle_mass_mapping: wp.array(dtype=wp.float32),
        particle_charge_mapping: wp.array(dtype=wp.float32),
        material_id: wp.array4d(dtype=wp.uint8),
        permiable_mapping: wp.array(dtype=wp.uint8),
        origin: tuple[float, float, float],
        spacing: tuple[float, float, float],
        dt: float,
        nr_ghost_cells: int = 1,
    ):
        # Launch kernel
        wp.launch(
            self._update_position,
            inputs=[
                particle_position,
                particle_velocity,
                particle_id,
                current_density,
                particle_mass_mapping,
                particle_charge_mapping,
                material_id,
                permiable_mapping,
                origin,
                spacing,
                dt,
                nr_ghost_cells,
            ],
            dim=particle_position.shape[1],
        )

        return particle_position


class DepositCharge(Operator):
    """
    Charge deposition operator
    """

    @wp.kernel
    def _deposit_charge(
        particle_position: wp.array2d(dtype=wp.float32),
        particle_id: wp.array2d(dtype=wp.uint8),
        charge_density: wp.array4d(dtype=wp.float32),
        particle_charge_mapping: wp.array(dtype=wp.float32),
        origin: wp.vec3f,
        spacing: wp.vec3f,
        nr_ghost_cells: wp.int32,
    ):
        # get particle index
        i = wp.tid()

        # Get particle position and id
        pos = wp.vec3f(
            particle_position[0, i], particle_position[1, i], particle_position[2, i]
        )
        pid = particle_id[0, i]

        # Get properties
        charge = particle_charge_mapping[wp.int32(pid)]

        # Get Ghost cell offset
        ghost_cell_offset = wp.vec3f(
            spacing[0] * wp.float32(nr_ghost_cells),
            spacing[1] * wp.float32(nr_ghost_cells),
            spacing[2] * wp.float32(nr_ghost_cells),
        )

        # Get origins for charge density
        origin_rho = (
            origin
            - wp.vec3f(spacing[0] / 2.0, spacing[1] / 2.0, spacing[2] / 2.0)
            - ghost_cell_offset
        )

        # Get lower cell index for charge density
        f_ijk_rho = wp.cw_div(pos - origin_rho, spacing) - wp.vec3f(0.5, 0.5, 0.5)
        ijk_rho = wp.vec3i(
            wp.int32(f_ijk_rho[0]), wp.int32(f_ijk_rho[1]), wp.int32(f_ijk_rho[2])
        )

        # Get position for each dimension
        cell_pos_x_lower = origin_rho[0] + (wp.float32(ijk_rho[0]) + 0.5) * spacing[0]
        cell_pos_y_lower = origin_rho[1] + (wp.float32(ijk_rho[1]) + 0.5) * spacing[1]
        cell_pos_z_lower = origin_rho[2] + (wp.float32(ijk_rho[2]) + 0.5) * spacing[2]
        cell_pos_x_upper = cell_pos_x_lower + spacing[0]
        cell_pos_y_upper = cell_pos_y_lower + spacing[1]
        cell_pos_z_upper = cell_pos_z_lower + spacing[2]

        # Get shape functions for each corner
        s_x_lower = linear_shape_function(pos[0], cell_pos_x_lower, spacing[0])
        s_y_lower = linear_shape_function(pos[1], cell_pos_y_lower, spacing[1])
        s_z_lower = linear_shape_function(pos[2], cell_pos_z_lower, spacing[2])
        s_x_upper = linear_shape_function(pos[0], cell_pos_x_upper, spacing[0])
        s_y_upper = linear_shape_function(pos[1], cell_pos_y_upper, spacing[1])
        s_z_upper = linear_shape_function(pos[2], cell_pos_z_upper, spacing[2])

        # Deposit charge
        wp.atomic_add(
            charge_density,
            0,
            wp.int32(ijk_rho[0]),
            wp.int32(ijk_rho[1]),
            wp.int32(ijk_rho[2]),
            charge * s_x_lower * s_y_lower * s_z_lower,
        )
        wp.atomic_add(
            charge_density,
            0,
            wp.int32(ijk_rho[0]) + 1,
            wp.int32(ijk_rho[1]),
            wp.int32(ijk_rho[2]),
            charge * s_x_upper * s_y_lower * s_z_lower,
        )
        wp.atomic_add(
            charge_density,
            0,
            wp.int32(ijk_rho[0]),
            wp.int32(ijk_rho[1]) + 1,
            wp.int32(ijk_rho[2]),
            charge * s_x_lower * s_y_upper * s_z_lower,
        )
        wp.atomic_add(
            charge_density,
            0,
            wp.int32(ijk_rho[0]) + 1,
            wp.int32(ijk_rho[1]) + 1,
            wp.int32(ijk_rho[2]),
            charge * s_x_upper * s_y_upper * s_z_lower,
        )
        wp.atomic_add(
            charge_density,
            0,
            wp.int32(ijk_rho[0]),
            wp.int32(ijk_rho[1]),
            wp.int32(ijk_rho[2]) + 1,
            charge * s_x_lower * s_y_lower * s_z_upper,
        )
        wp.atomic_add(
            charge_density,
            0,
            wp.int32(ijk_rho[0]) + 1,
            wp.int32(ijk_rho[1]),
            wp.int32(ijk_rho[2]) + 1,
            charge * s_x_upper * s_y_lower * s_z_upper,
        )
        wp.atomic_add(
            charge_density,
            0,
            wp.int32(ijk_rho[0]),
            wp.int32(ijk_rho[1]) + 1,
            wp.int32(ijk_rho[2]) + 1,
            charge * s_x_lower * s_y_upper * s_z_upper,
        )
        wp.atomic_add(
            charge_density,
            0,
            wp.int32(ijk_rho[0]) + 1,
            wp.int32(ijk_rho[1]) + 1,
            wp.int32(ijk_rho[2]) + 1,
            charge * s_x_upper * s_y_upper * s_z_upper,
        )

    def __call__(
        self,
        particle_position: wp.array2d(dtype=wp.float32),
        particle_id: wp.array2d(dtype=wp.uint8),
        charge_density: wp.array4d(dtype=wp.float32),
        particle_charge_mapping: wp.array(dtype=wp.float32),
        origin: tuple[float, float, float],
        spacing: tuple[float, float, float],
        nr_ghost_cells: int = 1,
    ):
        # Launch kernel
        wp.launch(
            self._deposit_charge,
            inputs=[
                particle_position,
                particle_id,
                charge_density,
                particle_charge_mapping,
                origin,
                spacing,
                nr_ghost_cells,
            ],
            dim=particle_position.shape[1],
        )

        return charge_density


class ChargeConservation(Operator):
    """
    Charge conservation operator
    """

    @wp.kernel
    def _charge_conservation(
        charge_density_0: wp.array4d(dtype=wp.float32),
        charge_density_1: wp.array4d(dtype=wp.float32),
        current_density: wp.array4d(dtype=wp.float32),
        charge_conservation: wp.array4d(dtype=wp.float32),
        spacing: wp.vec3f,
        dt: wp.float32,
        nr_ghost_cells: wp.int32,
    ):
        # get index
        i, j, k = wp.tid()

        # Skip ghost cells
        i += nr_ghost_cells
        j += nr_ghost_cells
        k += nr_ghost_cells

        # Get needed charge density
        rho_0 = charge_density_0[0, i, j, k]
        rho_1 = charge_density_1[0, i, j, k]

        # Get needed current density
        j_x_0 = current_density[0, i - 1, j, k]
        j_x_1 = current_density[0, i, j, k]
        j_y_0 = current_density[1, i, j - 1, k]
        j_y_1 = current_density[1, i, j, k]
        j_z_0 = current_density[2, i, j, k - 1]
        j_z_1 = current_density[2, i, j, k]

        # Get divergence of current density
        div_j_x = (j_x_1 - j_x_0) / spacing[0]
        div_j_y = (j_y_1 - j_y_0) / spacing[1]
        div_j_z = (j_z_1 - j_z_0) / spacing[2]
        div_j = div_j_x + div_j_y + div_j_z

        # Calculate charge conservation
        charge_conservation[0, i, j, k] = (rho_1 - rho_0) + dt * div_j

    def __call__(
        self,
        charge_density_0: wp.array4d(dtype=wp.float32),
        charge_density_1: wp.array4d(dtype=wp.float32),
        current_density: wp.array4d(dtype=wp.float32),
        charge_conservation: wp.array4d(dtype=wp.float32),
        spacing: tuple[float, float, float],
        dt: float,
        nr_ghost_cells: int = 1,
    ):
        # Launch kernel
        wp.launch(
            self._charge_conservation,
            inputs=[
                charge_density_0,
                charge_density_1,
                current_density,
                charge_conservation,
                spacing,
                dt,
                nr_ghost_cells,
            ],
            dim=current_density.shape[1:],
        )

        return charge_conservation


class InitializeParticles(Operator):

    _k = wp.constant(1.380649e-23)

    @wp.kernel
    def _set_particles(
        particle_position: wp.array2d(dtype=wp.float32),
        particle_velocity: wp.array2d(dtype=wp.float32),
        particle_id: wp.array2d(dtype=wp.uint8),
        particle_mass_mapping: wp.array(dtype=wp.float32),
        nr_particles: int,
        add_species_ids: wp.array(dtype=wp.uint8),
        mesh: wp.uint64,
        lower_bound: wp.vec3f,
        upper_bound: wp.vec3f,
        temperature: wp.float32,
    ):
        # Get particle index
        i = wp.tid()

        # Initialize random seed
        r = wp.rand_init(i)

        # Compute maximum distance to check
        max_distance = wp.sqrt(
            (upper_bound[0] - lower_bound[0]) ** 2.0
            + (upper_bound[1] - lower_bound[1]) ** 2.0
            + (upper_bound[2] - lower_bound[2]) ** 2.0
        )

        # Loop until particle is inside the domain
        counter = 0
        for _ in range(100):

            # Get random position
            x = lower_bound[0] + (upper_bound[0] - lower_bound[0]) * wp.randf(r)
            y = lower_bound[1] + (upper_bound[1] - lower_bound[1]) * wp.randf(r)
            z = lower_bound[2] + (upper_bound[2] - lower_bound[2]) * wp.randf(r)
            pos = wp.vec3f(x, y, z)

            # Evaluate distance to mesh
            face_index = int(0)
            face_u = float(0.0)
            face_v = float(0.0)
            sign = float(0.0)
            if (wp.mesh_query_point(mesh, pos, max_distance, sign, face_index, face_u, face_v)):
                p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
                delta = pos - p
                norm = wp.sqrt(wp.dot(delta, delta))
    
                # If particle is inside the mesh, break and add it
                if (sign < 0):

                    # Set particle position
                    for ii in range(add_species_ids.shape[0]):

                        # Get id of particle
                        id = wp.int32(add_species_ids[ii])

                        # Get sigma for random velocity
                        sigma = wp.sqrt(InitializeParticles._k * temperature / particle_mass_mapping[id])

                        # Get random velocity
                        vx = sigma * wp.randn(r) # Fix to be normal distribution
                        vy = sigma * wp.randn(r)
                        vz = sigma * wp.randn(r)

                        # Get particle index
                        particle_index = add_species_ids.shape[0] * i + ii + nr_particles

                        # Set particle position
                        particle_position[0, particle_index] = pos[0]
                        particle_position[1, particle_index] = pos[1]
                        particle_position[2, particle_index] = pos[2]

                        # Set particle velocity
                        particle_velocity[0, particle_index] = vx
                        particle_velocity[1, particle_index] = vy
                        particle_velocity[2, particle_index] = vz

                        # Set particle id
                        particle_id[0, particle_index] = wp.uint8(id)

                    # Break
                    break

    def __call__(
        self,
        particle_position: wp.array2d(dtype=wp.float32),
        particle_velocity: wp.array2d(dtype=wp.float32),
        particle_id: wp.array2d(dtype=wp.uint8),
        particle_mass_mapping: wp.array(dtype=wp.float32),
        nr_particles: int,
        nr_particles_add: int,
        add_species_ids: np.array,
        geometry: Compound,
        temperature: float,
        tolerance: float = 0.001,
        angular_tolerance: float = 0.1,
    ):

        # Export geometry to stl
        with tempfile.NamedTemporaryFile(suffix=".stl") as f:
            geometry.export_stl(
                f.name, tolerance=tolerance, angular_tolerance=angular_tolerance
            )
            mesh = np_mesh.Mesh.from_file(f.name)
            mesh_points = mesh.points.reshape(-1, 3)
            mesh_indices = np.arange(mesh_points.shape[0])
            mesh = wp.Mesh(
                points=wp.array(mesh_points, dtype=wp.vec3),
                indices=wp.array(mesh_indices, dtype=int),
            )

        # Get lower and upper bounds
        lower_bound = np.min(mesh_points, axis=0)
        upper_bound = np.max(mesh_points, axis=0)

        # Launch kernel
        wp.launch(
            self._set_particles,
            inputs=[
                particle_position,
                particle_velocity,
                particle_id,
                particle_mass_mapping,
                nr_particles,
                wp.from_numpy(add_species_ids, dtype=wp.uint8),
                mesh.id,
                lower_bound,
                upper_bound,
                temperature,
            ],
            dim=nr_particles_add,
        )

        return particle_position, particle_velocity, particle_id, nr_particles + nr_particles_add * len(add_species_ids)



class ComputeParticleEnergy(Operator):

    _k = wp.constant(1.380649e-23)

    @wp.kernel
    def _compute_particle_energy(
        total_energy: wp.array(dtype=wp.float32),
        particle_velocity: wp.array2d(dtype=wp.float32),
        particle_id: wp.array2d(dtype=wp.uint8),
        particle_mass_mapping: wp.array(dtype=wp.float32),
        nr_particles: int,
        spacing: wp.vec3f,
    ):
        # Get particle index
        i = wp.tid()

        # Get particle velocity and id
        vx = particle_velocity[0, i]
        vy = particle_velocity[1, i]
        vz = particle_velocity[2, i]
        pid = particle_id[0, i]

        # Get particle mass
        mass = particle_mass_mapping[wp.int32(pid)]

        # Compute kinetic energy
        vol = spacing[0] * spacing[1] * spacing[2]
        wp.atomic_add(total_energy, 0, 0.5 * vol * mass * (vx ** 2.0 + vy ** 2.0 + vz ** 2.0))

    def __call__(
        self,
        particle_velocity: wp.array2d(dtype=wp.float32),
        particle_id: wp.array2d(dtype=wp.uint8),
        particle_mass_mapping: wp.array(dtype=wp.float32),
        nr_particles: int,
        spacing: tuple[float, float, float],
    ):

        # Initialize total energy
        total_energy = wp.zeros(1, dtype=wp.float32)

        # Launch kernel
        wp.launch(
            self._compute_particle_energy,
            inputs=[
                total_energy,
                particle_velocity,
                particle_id,
                particle_mass_mapping,
                nr_particles,
                wp.vec3f(spacing[0], spacing[1], spacing[2]),
            ],
            dim=nr_particles,
        )

        return total_energy.numpy()[0]





# class ElectronEmission(Operator):
#    """
#    Electron emission operator
#    """
#
#    @wp.kernel
#    def _electron_emission(
#        material_id: wp.array4d(dtype=wp.uint8),
#        charge_density: wp.array4d(dtype=wp.float32),
#        counter: wp.array(dtype=wp.int32),
#        particle_charge: float,
#        particle_position_buffer: wp.array2d(dtype=wp.float32),
#        particle_velocity_buffer: wp.array2d(dtype=wp.float32),
#        particle_id_buffer: wp.array2d(dtype=wp.uint8),
#        particle_id: wp.uint8,
#        origin: wp.vec3f,
#        spacing: wp.vec3f,
#        dt: wp.float32,
#        nr_ghost_cells: wp.int32,
#    ):
#
#        # get material id index
#        i, j, k = wp.tid()
#
#        # Skip ghost cells
#        i += nr_ghost_cells
#        j += nr_ghost_cells
#        k += nr_ghost_cells
#
#        # Get material id
#        mat_id = material_id[0, i, j, k]
#
#        # Run through faces
#        for d in range(3):
#
#            # Get neighbor material id
#            if d == 0:
#                interface_mat_id = material_id[0, i-1, j, k]
#            elif d == 1:
#                interface_mat_id = material_id[0, i, j-1, k]
#            else:
#                interface_mat_id = material_id[0, i, j, k-1]
#
#            # Check if interface is between a vacuum and other material
#            if (mat_id == wp.uint8(0) and interface_mat_id != wp.uint8(0)) or (mat_id != wp.uint8(0) and interface_mat_id == wp.uint8(0)):
#
#                # Get ghost cell offset
#                ghost_cell_offset = wp.vec3f(spacing[0] * wp.float32(nr_ghost_cells), spacing[1] * wp.float32(nr_ghost_cells), spacing[2] * wp.float32(nr_ghost_cells))
#
#                # Get origins for charge density
#                origin_rho = origin - wp.vec3f(spacing[0]/2.0, spacing[1]/2.0, spacing[2]/2.0) - ghost_cell_offset
#
#                # Get lower cell index for charge density
#                f_ijk_rho = wp.vec3i(i, j, k)
#                ijk_rho = wp.vec3i(wp.int32(f_ijk_rho[0]), wp.int32(f_ijk_rho[1]), wp.int32(f_ijk_rho[2]))
#
#                # Get position for each dimension
#                cell_pos_x_lower = origin_rho[0] + (wp.float32(ijk_rho[0]) + 0.5) * spacing[0]
#                cell_pos_y_lower = origin_rho[1] + (wp.float32(ijk_rho[1]) + 0.5) * spacing[1]
#                cell_pos_z_lower = origin_rho[2] + (wp.float32(ijk_rho[2]) + 0.5) * spacing[2]
#                cell_pos_x_upper = cell_pos_x_lower + spacing[0]
#                cell_pos_y_upper = cell_pos_y_lower + spacing[1]
#                cell_pos_z_upper = cell_pos_z_lower + spacing[2]
#
#                # Get shape functions for each corner
#                s_x_lower = linear_shape_function(cell_pos_x_lower, cell_pos_x_lower, spacing[0])
#                s_y_lower = linear_shape_function(cell_pos_y_lower, cell_pos_y_lower, spacing[1])
#                s_z_lower = linear_shape_function(cell_pos_z_lower, cell_pos_z_lower, spacing[2])
#                s_x_upper = linear_shape_function(cell_pos_x_upper, cell_pos_x_upper, spacing[0])
#                s_y_upper = linear_shape_function(cell_pos_y_upper, cell_pos_y_upper, spacing[1])
#                s_z_upper = linear_shape_function(cell_pos_z_upper, cell_pos_z_upper, spacing[2])
#
#                # Get charge density
#                rho = (
#                    charge_density[0, ijk_rho[0], ijk_rho
#
#
