import warp as wp

from pumpkin_pulse.operator.operator import Operator

wp.init()

@wp.struct
class EMFields:
    # electric field components
    ex : wp.array3d(dtype=wp.float32)
    ey : wp.array3d(dtype=wp.float32)
    ez : wp.array3d(dtype=wp.float32)

    # magnetic field components
    bx : wp.array3d(dtype=wp.float32)
    by : wp.array3d(dtype=wp.float32)
    bz : wp.array3d(dtype=wp.float32)

    # current density components
    jx : wp.array3d(dtype=wp.float32)
    jy : wp.array3d(dtype=wp.float32)
    jz : wp.array3d(dtype=wp.float32)

    # charge density
    rho : wp.array3d(dtype=wp.float32)

    # potential
    phi : wp.array3d(dtype=wp.float32)

    # Temperature
    temp : wp.array3d(dtype=wp.float32)

    # id information
    id: wp.array3d(dtype=wp.uint8)

    # id information mapping
    eps_mapping: wp.array(dtype=wp.float32)
    mu_mapping: wp.array(dtype=wp.float32)
    sigma_mapping: wp.array(dtype=wp.float32)
    specific_heat_mapping: wp.array(dtype=wp.float32)

    # Grid information
    spacing: wp.vec3
    shape: wp.vec3i
    nr_ghost_cells: wp.int32

    # origins for all the fields
    ex_origin: wp.vec3
    ey_origin: wp.vec3
    ez_origin: wp.vec3
    bx_origin: wp.vec3
    by_origin: wp.vec3
    bz_origin: wp.vec3
    jx_origin: wp.vec3
    jy_origin: wp.vec3
    jz_origin: wp.vec3
    rho_origin: wp.vec3
    phi_origin: wp.vec3
    temp_origin: wp.vec3
    id_origin: wp.vec3

    # Time for the fields
    e_time: wp.float32
    b_time: wp.float32
    j_time: wp.float32
    rho_time: wp.float32
    phi_time: wp.float32
    temp_time: wp.float32


class AllocateEMFields(Operator):

    def __call__(
        self,
        origin: wp.vec3,
        spacing: wp.vec3,
        shape: wp.vec3i,
        nr_id_types: wp.int32,
        nr_ghost_cells: wp.int32
    ):

        # Get the shape with ghost cells
        shape_with_ghost = [s + 2 * nr_ghost_cells for s in shape]

        # Allocate the fields
        em_fields = EMFields()

        # Allocate the electric field components
        em_fields.ex = wp.zeros(shape_with_ghost, dtype=wp.float32)
        em_fields.ey = wp.zeros(shape_with_ghost, dtype=wp.float32)
        em_fields.ez = wp.zeros(shape_with_ghost, dtype=wp.float32)

        # Allocate the magnetic field components
        em_fields.bx = wp.zeros(shape_with_ghost, dtype=wp.float32)
        em_fields.by = wp.zeros(shape_with_ghost, dtype=wp.float32)
        em_fields.bz = wp.zeros(shape_with_ghost, dtype=wp.float32)

        # Allocate the current density components
        em_fields.jx = wp.zeros(shape_with_ghost, dtype=wp.float32)
        em_fields.jy = wp.zeros(shape_with_ghost, dtype=wp.float32)
        em_fields.jz = wp.zeros(shape_with_ghost, dtype=wp.float32)

        # Allocate the charge density
        em_fields.rho = wp.zeros(shape_with_ghost, dtype=wp.float32)

        # Allocate the potential
        em_fields.phi = wp.zeros(shape_with_ghost, dtype=wp.float32)

        # Allocate the temperature
        em_fields.temp = wp.zeros(shape_with_ghost, dtype=wp.float32)

        # Allocate the id information
        em_fields.id = wp.zeros(shape_with_ghost, dtype=wp.uint8)

        # Allocate the id information mapping
        em_fields.eps_mapping = wp.zeros(nr_id_types, dtype=wp.float32)
        em_fields.mu_mapping = wp.zeros(nr_id_types, dtype=wp.float32)
        em_fields.sigma_mapping = wp.zeros(nr_id_types, dtype=wp.float32)
        em_fields.specific_heat_mapping = wp.zeros(nr_id_types, dtype=wp.float32)

        # Set the grid information
        em_fields.spacing = wp.vec3(spacing)
        em_fields.shape = wp.vec3i(shape)
        em_fields.nr_ghost_cells = nr_ghost_cells

        # Set the origins for all the fields (Yee grid)
        em_fields.ex_origin = wp.vec3([origin[0], origin[1] - 0.5 * spacing[1], origin[2] - 0.5 * spacing[2]])
        em_fields.ey_origin = wp.vec3([origin[0] - 0.5 * spacing[0], origin[1], origin[2] - 0.5 * spacing[2]])
        em_fields.ez_origin = wp.vec3([origin[0] - 0.5 * spacing[0], origin[1] - 0.5 * spacing[1], origin[2]])
        em_fields.bx_origin = wp.vec3([origin[0] - 0.5 * spacing[0], origin[1], origin[2]])
        em_fields.by_origin = wp.vec3([origin[0], origin[1] - 0.5 * spacing[1], origin[2]])
        em_fields.bz_origin = wp.vec3([origin[0], origin[1], origin[2] - 0.5 * spacing[2]])
        em_fields.jx_origin = em_fields.ex_origin
        em_fields.jy_origin = em_fields.ey_origin
        em_fields.jz_origin = em_fields.ez_origin
        em_fields.rho_origin = wp.vec3([origin[0] - 0.5 * spacing[0], origin[1] - 0.5 * spacing[1], origin[2] - 0.5 * spacing[2]])
        em_fields.phi_origin = em_fields.rho_origin
        em_fields.temp_origin = wp.vec3([origin[0], origin[1], origin[2]])
        em_fields.id_origin = em_fields.temp_origin

        return em_fields

if __name__ == '__main__':
    allocate_em_fields = AllocateEMFields()
    em_fields = allocate_em_fields((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (10, 10, 10), 2, 1)
