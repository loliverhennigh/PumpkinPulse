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
    nr_particles = 130000000
    origin = (-6.0, -3.0, -3.0)
    spacing = (0.05, 0.05, 0.05)
    shape = (240, 120, 120)
    nr_ghost_cells = 0
    temperature = 1000.0
    velocity = 5000.0

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
    stl_file = "files/X-71_super_space_shuttle.stl"
    mesh = stl_to_mesh(stl_file)
    material_properties = mesh_material_property_setter(
        material_properties=material_properties,
        mesh=mesh,
        id_number=1,
    )

    # Save solid meshes
    marching_cube_saver(
        material_properties,
        filename="shuttle_solid.vtk",
    )

    # Set box
    inlet_box = Rectangle(-5.9, -5.9)
    inlet_box = extrude(inlet_box, 0.1)
    inlet_box = Rotation(0.0, 90.0, 0.0) * inlet_box
    inlet_box = Location((-5.9, 0.0, 0.0)) * inlet_box
    inlet_box_mesh = build123d_to_mesh(inlet_box)
    nr_particles_per_cell = 12
    particles = particle_injector(
        particles,
        mesh=inlet_box_mesh,
        nr_particles_per_cell=nr_particles_per_cell,
        temperature=temperature,
        mean_velocity=(velocity, 0.0, 0.0),
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
                filename=f"shuttle_field_{str(i).zfill(5)}.vtk",
            )
        particles = particle_injector(
            particles,
            mesh=inlet_box_mesh,
            nr_particles_per_cell=nr_particles_per_cell,
            temperature=temperature,
            mean_velocity=(velocity, 0.0, 0.0),
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
