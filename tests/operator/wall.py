import pytest
from build123d import Rectangle, extrude, Location
import warp as wp
import numpy as np
import matplotlib.pyplot as plt

from pumpkin_pulse.operator.allocator import AllocateParticles
from pumpkin_pulse.operator.allocator import MaterialProperties
from pumpkin_pulse.operator.particle_injector import ParticleInjector
from pumpkin_pulse.operator.geometry import Build123DToMesh
from pumpkin_pulse.operator.pusher import BasicPusher
from pumpkin_pulse.operator.io import ParticleSaver

def test_basic_pusher():

    # Make operators
    allocate_particles = AllocateParticles()
    build123d_to_mesh = Build123DToMesh()
    particle_injector = ParticleInjector()
    pusher = BasicPusher()
    saver = ParticleSaver()

    # Allocate particles
    particles = allocate_particles(
        nr_particles=100000,
        origin=(-10.0, -10.0, -10.0),
        spacing=(1.0, 1.0, 1.0),
        shape=(20, 20, 20),
        nr_id_types=2,
        nr_ghost_cells=0,
    )

    # Set mass mapping (mass of proton)
    np_mass_mapping = np.ones((3,)) * 1.6726219e-27
    particles.mass_mapping = wp.from_numpy(np_mass_mapping, dtype=wp.float32)

    # Make mesh
    length = 3.0 # (-1.5, 1.5)
    cube = extrude(Rectangle(length, length), length)
    cube = Location((0.0, 0.0, -length / 2.0)) * cube
    mesh = build123d_to_mesh(cube)

    # Inject particles
    nr_particles_per_cell = 1000
    injected_particles = particle_injector(
        particles,
        mesh=mesh,
        nr_particles_per_cell=nr_particles_per_cell,
        add_species_ids=wp.array([1, 2], dtype=wp.uint8),
        temperature=300.0,
    )

    # Push particles
    for i in range(10):
        particles = pusher(
            particles,
            dt=5e-6,
        )
        saver(
            particles,
            filename=f"test_basic_pusher_{i}.vtk",
            save_velocity=True,
            save_index=True,
        )

    # Check that all particles are within the mesh
    np_pos = particles.position[:particles.nr_particles.numpy()[0]].numpy()
    assert np.all(np_pos >= (-1.5, -1.5, -1.5))
    assert np.all(np_pos <= (1.5, 1.5, 1.5))
