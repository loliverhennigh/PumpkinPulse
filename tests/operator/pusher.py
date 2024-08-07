import pytest
from build123d import Rectangle, extrude, Sphere, Location, Circle, Rotation
import warp as wp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from pumpkin_pulse.operator.allocator import ParticleAllocator, MaterialPropertyAllocator
from pumpkin_pulse.operator.material_property_setter import MeshMaterialPropertySetter
from pumpkin_pulse.operator.particle_injector import ParticleInjector
from pumpkin_pulse.operator.mesh import Build123DToMesh, StlToMesh
from pumpkin_pulse.operator.pusher import NeutralPusher
from pumpkin_pulse.operator.saver import ParticleSaver, MaterialPropertiesSaver, MarchingCubeSaver, FieldSaver
from pumpkin_pulse.operator.collision.hard_sphere import HardSphereCollision

if __name__ == "__main__":

    # Make operators
    particle_allocator = ParticleAllocator()
    material_property_allocator = MaterialPropertyAllocator()
    mesh_material_property_setter = MeshMaterialPropertySetter()
    build123d_to_mesh = Build123DToMesh()
    particle_injector = ParticleInjector()
    pusher = NeutralPusher()
    particle_saver = ParticleSaver()
    material_properties_saver = MaterialPropertiesSaver()
    marching_cube_saver = MarchingCubeSaver()
    hard_sphere_collision = HardSphereCollision()
    field_saver = FieldSaver()
    stl_to_mesh = StlToMesh()

    # Parameters
    nr_particles = 100000000
    origin = (-5.0, -5.0, -5.0)
    spacing = (0.05, 0.05, 0.05)
    shape = (200, 200, 200)
    nr_ghost_cells = 0
    cylinder_radius = 0.2
    cylinder_height = 1.5
    cylinder_thickness = 0.3
    temperature = 10000.0

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
    sphere = Location((1.5, 0.0, 0.0)) * Sphere(1.5)
    sphere_mesh = build123d_to_mesh(sphere)
    material_properties = mesh_material_property_setter(
        material_properties=material_properties,
        mesh=sphere_mesh,
        id_number=1,
    )

    # Set solid cylinder in material properties
    outer_circle = Circle(cylinder_radius+cylinder_thickness)
    inner_circle = Circle(cylinder_radius)
    cylinder = extrude(outer_circle - inner_circle, cylinder_height)
    cylinder = Rotation(0.0, 90.0, 0.0) * cylinder
    cylinder = Location((-5.0, 0.0, 0.0)) * cylinder
    cylinder_mesh = build123d_to_mesh(cylinder)
    material_properties = mesh_material_property_setter(
        material_properties=material_properties,
        mesh=cylinder_mesh,
        id_number=1,
    )

    # Save solid meshes
    marching_cube_saver(
        material_properties,
        filename="test_basic_pusher_sphere.vtk",
    )

    # Set circle of particles
    inlet_circle = Circle(cylinder_radius-0.1)
    inlet_circle = extrude(inlet_circle, 0.1)
    inlet_circle = Rotation(0.0, 90.0, 0.0) * inlet_circle
    inlet_circle = Location((-4.9, 0.0, 0.0)) * inlet_circle
    inlet_circle_mesh = build123d_to_mesh(inlet_circle)
    nr_particles_per_cell = 15000
    particles = particle_injector(
        particles,
        mesh=inlet_circle_mesh,
        nr_particles_per_cell=nr_particles_per_cell,
        temperature=temperature,
        mean_velocity=(0.0, 0.0, 0.0),
    )
    
    particles = pusher(
        particles,
        material_properties=material_properties,
        dt=1.0e-6,
    )
    wp.synchronize()
 
    # Push particles
    tic = time.time()
    nr_steps = 700
    for i in tqdm(range(nr_steps)):
        print(f"Step {i}")
        print(f"Number of particles: {particles.nr_particles.numpy()[0]}")
        if i % 1 == 0:
            #particle_saver(
            #    particles,
            #    filename=f"test_basic_pusher_{str(i).zfill(5)}.vtk",
            #    save_velocity=True,
            #    save_index=True,
            #)
            field_saver(
                particles.cell_particle_mapping_buffer,
                filename=f"test_basic_pusher_field_{str(i).zfill(5)}.vtk",
            )
        particles = particle_injector(
            particles,
            mesh=inlet_circle_mesh,
            nr_particles_per_cell=nr_particles_per_cell,
            temperature=temperature,
            mean_velocity=(0.0, 0.0, 0.0),
        )
        particles = pusher(
            particles,
            material_properties=material_properties,
            dt=5.0e-6,
        )
        particles = hard_sphere_collision(
            particles,
            dt=5.0e-6,
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
