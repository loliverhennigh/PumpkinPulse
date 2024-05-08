import pytest

from pumpkin_pulse.operator.allocator import EMFieldAllocator, ParticleAllocator, MaterialPropertyAllocator

def test_em_field_allocator():
    em_field_allocator = EMFieldAllocator()
    em_field_allocator(
        origin=(0, 0, 0),
        spacing=(1, 1, 1),
        shape=(10, 10, 10),
        nr_ghost_cells=1,
    )

def test_particle_allocator():
    particle_allocator = ParticleAllocator()
    particle_allocator(
        nr_particles=100,
        charge=1.0,
        mass=1.0,
        origin=(0, 0, 0),
        spacing=(1, 1, 1),
        shape=(10, 10, 10),
        nr_ghost_cells=1,
    )

def test_material_property_allocator():
    material_property_allocator = MaterialPropertyAllocator()
    material_property_allocator(
        nr_materials=3,
        origin=(0, 0, 0),
        spacing=(1, 1, 1),
        shape=(10, 10, 10),
        nr_ghost_cells=1,
    )
