# Purpose: Base class for constructing the fields and transforming the operators between them

import warp as wp
from typing import Union

from pumpkin_pulse.data.field import Field, Fieldint32
from pumpkin_pulse.data.particles import Particles
from pumpkin_pulse.operator.operator import Operator

class Constructor:
    """
    Base class for constructing the fields and transforming the operators
    """

    def __init__(
        self,
        shape: tuple,
        origin: tuple,
        spacing: tuple,
    ):
        self.shape = shape
        self.origin = origin
        self.spacing = spacing

    def create_field(
        self,
        dtype: type,
        cardinality: int,
        offset: tuple = None,
        shape: tuple = None,
        ordering: Union[int, str] = "SoA",
    ):

        # Get ordering as integer
        if not isinstance(ordering, int):
            if ordering == "SoA":
                ordering = 0
            elif ordering == "AoS":
                ordering = 1
            else:
                raise ValueError(f"Unknown ordering: {ordering}")

        # Get offset and shape
        if offset is None:
            offset = (0, 0, 0)
        if shape is None:
            shape = self.shape

        # Allocate the field
        field = Field(dtype)()

        # Set the field properties
        if ordering == 0:
            field.data = wp.zeros([cardinality] + list(shape), dtype=dtype)
        elif ordering == 1:
            field.data = wp.zeros(list(shape) + [cardinality], dtype=dtype)
        else:
            raise ValueError(f"Unknown ordering: {ordering}")
        field.cardinality = wp.int32(cardinality)
        field.shape = wp.vec3i(shape)
        field.origin = wp.vec3(self.origin)
        field.spacing = wp.vec3(self.spacing)
        field.offset = wp.vec3i(offset)
        field.ordering = wp.uint8(ordering)

        return field

    def create_particles(
        self,
        num_particles: int,
        mass: float,
        charge: float,
        volume: float,
        shape: tuple,
        origin: tuple,
        spacing: tuple,
    ):
        # Allocate the particles
        particles = Particles()

        # Set the particle properties
        particles.position = wp.zeros(num_particles, dtype=wp.vec3)
        particles.momentum = wp.zeros(num_particles, dtype=wp.vec3)
        particles.weight = wp.zeros(num_particles, dtype=wp.float32)
        particles.kill = wp.zeros(num_particles, dtype=wp.uint8)
        particles.num_particles = wp.zeros(1, dtype=wp.int32)
        particles.weight = wp.float32(weight)
        particles.charge = wp.float32(charge)

        # Set grid indexing information
        particles.grid_index = Fieldint32()
        particles.grid_index.data = wp.zeros([1] + list(shape), dtype=wp.int32)
        particles.grid_index.cardinality = wp.int32(1)
        particles.grid_index.shape = wp.vec3i(shape)
        particles.grid_index.origin = wp.vec3(origin)
        particles.grid_index.spacing = wp.vec3(spacing)

        return particles

    def transform_operator(self, operator: Operator):
        return operator
