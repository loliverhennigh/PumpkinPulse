import pytest
from build123d import Rectangle, extrude, Sphere, Location
import warp as wp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from pumpkin_pulse.operator.allocator import ParticleAllocator, MaterialPropertyAllocator
from pumpkin_pulse.operator.material_property_setter import MeshMaterialPropertySetter
from pumpkin_pulse.operator.particle_injector import ParticleInjector
from pumpkin_pulse.operator.mesh import Build123DToMesh
from pumpkin_pulse.operator.pusher import NeutralPusher
from pumpkin_pulse.operator.saver import ParticleSaver, MaterialPropertiesSaver

def test_basic_pusher():

    # Make operators
    particle_allocator = ParticleAllocator()
    material_property_allocator = MaterialPropertyAllocator()
    mesh_material_property_setter = MeshMaterialPropertySetter()
    build123d_to_mesh = Build123DToMesh()
    particle_injector = ParticleInjector()
    pusher = NeutralPusher()
    particle_saver = ParticleSaver()
    material_properties_saver = MaterialPropertiesSaver()

    # Parameters
    nr_particles = 100000000
    origin = (-10.0, -10.0, -10.0)
    spacing = (0.1, 0.1, 0.1)
    shape = (200, 200, 200)
    nr_ghost_cells = 0

    # Allocate particles
    particles = particle_allocator(
        nr_particles=nr_particles,
        origin=origin,
        spacing=spacing,
        shape=shape,
        nr_ghost_cells=nr_ghost_cells,
        mass=1.67e-27, # mass of proton in kg
        charge=0.0,
        weight=1.0,
    )

    # Allocate material properties
    material_properties = material_property_allocator(
        nr_materials=2,
        eps_mapping=np.array([1.0, 1.0]),
        mu_mapping=np.array([1.0, 1.0]),
        sigma_mapping=np.array([0.0, 0.0]),
        specific_heat_mapping=np.array([1.0, 1.0]),
        density_mapping=np.array([1.0, 1.0]),
        thermal_conductivity_mapping=np.array([1.0, 1.0]),
        solid_fraction_mapping=np.array([0.0, 1.0]),
        solid_type_mapping=np.array([0, 0]),
        origin=origin,
        spacing=spacing,
        shape=shape,
        nr_ghost_cells=nr_ghost_cells,
    )

    # Set solid sphere in material properties
    sphere = Location((2.5, 0.0, 0.0)) * Sphere(2.5)
    sphere_mesh = build123d_to_mesh(sphere)
    material_properties = mesh_material_property_setter(
        material_properties=material_properties,
        mesh=sphere_mesh,
        id_number=1,
    )
    material_properties_saver(
        material_properties,
        filename="test_basic_pusher_material_properties.vtk",
    )

    # Set cube of particles
    cube = extrude(Rectangle(4*spacing[0], 4*spacing[1]), 4*spacing[2])
    cube = Location((-2.5, 0.0, -spacing[2]/2.0)) * cube
    cube_mesh = build123d_to_mesh(cube)
    nr_particles_per_cell = 100000
    particles = particle_injector(
        particles,
        mesh=cube_mesh,
        nr_particles_per_cell=nr_particles_per_cell,
        temperature=5000.0,
        mean_velocity=(2000.0, 0.0, 0.0),
    )
    
    particles = pusher(
        particles,
        material_properties=material_properties,
        dt=1.0e-6,
    )
    wp.synchronize()
 
    # Push particles
    tic = time.time()
    nr_steps = 200
    for i in tqdm(range(nr_steps)):
        print(f"Step {i}")
        particle_saver(
            particles,
            filename=f"test_basic_pusher_{str(i).zfill(5)}.vtk",
            save_velocity=True,
            save_index=True,
        )
        particles = pusher(
            particles,
            material_properties=material_properties,
            dt=1.0e-5,
        )
 
        ## Check if any particles are NaN
        #if np.any(np.isnan(np.array([s[0] for s in particles.data[:particles.nr_particles.numpy()[0]].numpy()]))):
        #    print("NaN detected")
        #    raise ValueError
        #    exit()
    wp.synchronize()
    toc = time.time()
    nr_particles = particles.nr_particles.numpy()[0]
    particle_size = 4 * 6 + 1 # 4 bytes for each float, 6 floats for position and velocity, 1 for index
    print(f"Time: {toc-tic}")
    print(f"Million Particles: {nr_particles/1.0e6}")
    print(f"MUPS: {nr_particles * nr_steps / (toc-tic)/1.0e6}")
    print(f"GB/s: {nr_particles * nr_steps * particle_size / (toc-tic)/1.0e9}")

    exit()

    ## Check that all particles are within the mesh
    #np_pos = particles.position[:particles.nr_particles.numpy()[0]].numpy()
    #assert np.all(np_pos >= (-1.5, -1.5, -1.5))
    #assert np.all(np_pos <= (1.5, 1.5, 1.5))
