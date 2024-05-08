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
