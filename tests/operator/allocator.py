import pytest

import numpy as np

from pumpkin_pulse.operator.allocator import EMFieldAllocator, ParticleAllocator, MaterialPropertyAllocator

#def test_em_field_allocator():
#    em_field_allocator = EMFieldAllocator()
#    em_field_allocator(
#        origin=(0, 0, 0),
#        spacing=(1, 1, 1),
#        shape=(10, 10, 10),
#        nr_ghost_cells=1,
#    )

def test_particle_allocator():
    particle_allocator = ParticleAllocator()
    particle_allocator(
        nr_particles=100,
        weight=1.0,
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
        nr_materials=2,
        eps_mapping=np.array([1.0, 1.0]),
        mu_mapping=np.array([1.0, 1.0]),
        sigma_mapping=np.array([0.0, 0.0]),
        specific_heat_mapping=np.array([1.0, 1.0]),
        density_mapping=np.array([1.0, 1.0]),
        thermal_conductivity_mapping=np.array([1.0, 1.0]),
        solid_fraction_mapping=np.array([0.0, 1.0]),
        solid_type_mapping=np.array([0, 2]),
        origin=(0, 0, 0),
        spacing=(1, 1, 1),
        shape=(10, 10, 10),
        nr_ghost_cells=1,
    )
