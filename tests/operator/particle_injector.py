import pytest
from build123d import Rectangle, extrude, Location
import warp as wp
import numpy as np
import matplotlib.pyplot as plt

from pumpkin_pulse.operator.allocator import ParticleAllocator
from pumpkin_pulse.operator.particle_injector import ParticleInjector
from pumpkin_pulse.operator.mesh import Build123DToMesh

def test_particle_injector():

    # Make operators
    particle_allocator = ParticleAllocator()
    build123d_to_mesh = Build123DToMesh()
    particle_injector = ParticleInjector()

    # Allocate particles
    particles = particle_allocator(
        nr_particles=1000,
        origin=(-2.0, -2.0, -2.0),
        spacing=(1.0, 1.0, 1.0),
        shape=(4, 4, 4),
        nr_ghost_cells=0,
        charge=1.0,
        mass=1.0,
    )

    # Make mesh
    length = 3.0 # (-1.5, 1.5)
    cube = extrude(Rectangle(length, length), length)
    cube = Location((0.0, 0.0, -length / 2.0)) * cube
    mesh = build123d_to_mesh(cube)

    # Inject particles
    nr_particles_per_cell = 1
    injected_particles = particle_injector(
        particles,
        mesh=mesh,
        nr_particles_per_cell=nr_particles_per_cell,
        temperature=300.0,
        mean_velocity=(100.0, 0.0, 0.0),
    )

    # Check that the number of particles is correct
    assert particles.nr_particles.numpy()[0] == 3 * 3 * 3 * nr_particles_per_cell

    # Check that all particles are within the mesh
    np_data = particles.data[:particles.nr_particles.numpy()[0]].numpy()
    np_pos = np.array([np_data[i][0] for i in range(len(np_data))])
    assert np.all(np_pos >= (-1.5, -1.5, -1.5))
    assert np.all(np_pos <= (1.5, 1.5, 1.5))
