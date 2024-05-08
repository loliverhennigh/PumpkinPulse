# Base pusher class for pushing particles in time

import warp as wp

from pumpkin_pulse.struct.particles import Particles, PosMom
from pumpkin_pulse.struct.material_properties import MaterialProperties
from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.operator.sort.sort import ParticleSorter

class Pusher(Operator):
    sort_particles = ParticleSorter()

    def __init__(
        self,
        boundary_conditions: "periodic",
    ):

        # Set boundary condition function
        if boundary_conditions == "periodic":
            self.apply_boundary_conditions = Pusher.apply_periodic_boundary_conditions
        else:
            raise ValueError(
                f"Boundary conditions {boundary_conditions} not supported by NeutralPusher"
            )

    @wp.func
    def apply_periodic_boundary_conditions(
        pos: wp.vec3,
        origin: wp.vec3,
        spacing: wp.vec3,
        shape: wp.vec3i,
        nr_ghost_cells: wp.int32,
    ):

        # Check if particle is outside of domain
        # X-direction
        if pos[0] < origin[0]:
            pos[0] += wp.float32(shape[0] - 2 * nr_ghost_cells) * spacing[0]
        elif pos[0] >= origin[0] + wp.float32(shape[0] - 2 * nr_ghost_cells) * spacing[0]:
            pos[0] -= wp.float32(shape[0] - 2 * nr_ghost_cells) * spacing[0]

        # Y-direction
        if pos[1] < origin[1]:
            pos[1] += wp.float32(shape[1] - 2 * nr_ghost_cells) * spacing[1]
        elif pos[1] >= origin[1] + wp.float32(shape[1] - 2 * nr_ghost_cells) * spacing[1]:
            pos[1] -= wp.float32(shape[1] - 2 * nr_ghost_cells) * spacing[1]

        # Z-direction
        if pos[2] < origin[2]:
            pos[2] += wp.float32(shape[2] - 2 * nr_ghost_cells) * spacing[2]
        elif pos[2] >= origin[2] + wp.float32(shape[2] - 2 * nr_ghost_cells) * spacing[2]:
            pos[2] -= wp.float32(shape[2] - 2 * nr_ghost_cells) * spacing[2]

        return pos
