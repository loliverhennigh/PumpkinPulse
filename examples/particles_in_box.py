# Simple example of voxelizing a build123d geometry

import warp as wp
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyvista as pv

wp.init()

from dense_plasma_focus.operator.operator import Operator
from dense_plasma_focus.operator.electromagnetism.electromagnetism import ElectricFieldUpdate, MagneticFieldUpdate
from dense_plasma_focus.operator.pic.particle_pusher import BorisVelocityUpdate, PositionUpdate, DepositCharge, ChargeConservation

def plot_solution(
    particle_position,
    particle_velocity,
    current_density,
    charge_conservation,
    charge_density_0,
    charge_density_1,
    save_dir,
    step,
    origin,
    spacing
):

    # Get charge density and current density
    np_current_density = np.sum(current_density[0, :, :, :].numpy(), axis=-1)
    np_charge_density_0 = np.sum(charge_density_0[0, :, :, :].numpy(), axis=-1)
    np_charge_density_1 = np.sum(charge_density_1[0, :, :, :].numpy(), axis=-1)
    np_charge_conservation = np.sum(charge_conservation[0, :, :, :].numpy(), axis=-1)

    # Plot all side by side with colorbar
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    im0 = ax[0].imshow(np_current_density, cmap="viridis", origin="lower")
    ax[0].set_title("Current Density")
    fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(np_charge_density_0, cmap="viridis", origin="lower")
    ax[1].set_title("Charge Density")
    fig.colorbar(im1, ax=ax[1])
    im2 = ax[2].imshow(np_charge_conservation, cmap="viridis", origin="lower")
    ax[2].set_title("Charge Conservation")
    fig.colorbar(im2, ax=ax[2])
    plt.suptitle(f"Step {step}")

    plt.savefig(f"{save_dir}/step_{str(step).zfill(4)}.png")
    plt.close()


    ## Plot particle position on x-y plane
    #np_particle_position = particle_position.numpy()
    #np_particle_velocity = particle_velocity.numpy()
    #np_current_density = current_density[:, :, :, :].numpy()
    #np_current_density = np.sum(np_current_density, axis=-1)
    ##np_current_density = np.sqrt(np_current_density[0, :, :]**2 + np_current_density[1, :, :]**2 + np_current_density[2, :, :]**2)
    #np_current_density = np_current_density[0, :, :]
    #np_current_density = np_current_density[1:-1, 1:-1]

    ## Plot current density
    #plt.figure()
    #extent = np.array([origin[0] - spacing[0]/2.0, origin[0] + spacing[0] * np_current_density.shape[0] - spacing[0]/2.0, origin[1] - spacing[1]/2.0, origin[1] + spacing[1] * np_current_density.shape[1] - spacing[1]/2.0])
    #plt.imshow(np_current_density, extent=extent, cmap="jet", origin="lower")
    #plt.colorbar()
    #plt.scatter(np_particle_position[1, :], np_particle_position[0, :], c=np_particle_velocity[0, :], cmap="coolwarm")
    #plt.title(f"Step {step}")
    #plt.savefig(f"{save_dir}/current_density_{str(step).zfill(4)}.png")
    #plt.close()


def save_vtk(particle_position, particle_id, charge_density, save_dir, step, origin, spacing):

    # Create point cloud
    grid = pv.PolyData(particle_position.numpy().T)
    grid["particle_id"] = particle_id.numpy().T
    grid.save(f"{save_dir}/step_{str(step).zfill(4)}.vtk")

    # Create grid
    grid = pv.ImageData()
    grid.dimensions = [charge_density.shape[3], charge_density.shape[2], charge_density.shape[1]]
    grid.origin = origin
    grid.spacing = spacing

    # Get data
    np_charge_density = charge_density.numpy().flatten('F')

    # Add data
    grid.point_data["charge_density"] = np_charge_density

    # Save grid
    grid.save(f"{save_dir}/charge_density_{str(step).zfill(4)}.vtk")

times = []
kinetic_energys = []
magnetic_energys = []
electric_energys = []
total_energys = []
def compute_energy(
    particle_velocity,
    electric_field,
    magnetic_field,
    eps,
    mu,
    mass,
    dx,
    t,
):
    # Get arrays
    np_particle_velocity = particle_velocity.numpy()
    np_electric_field = electric_field.numpy()
    np_magnetic_field = magnetic_field.numpy()

    # Compute kinetic energy
    kinetic_energy = 0.5 * mass * dx**3 * np.sum(np_particle_velocity**2)

    # Compute electric energy
    electric_energy = 0.5 * eps * dx**3 * np.sum(np_electric_field**2)

    # Compute magnetic energy
    magnetic_energy = 0.5 * (1.0 / mu) * dx**3 * np.sum(np_magnetic_field**2)

    # Total energy
    total_energy = kinetic_energy + electric_energy + magnetic_energy

    # Append to list
    times.append(t)
    kinetic_energys.append(kinetic_energy)
    electric_energys.append(electric_energy)
    magnetic_energys.append(magnetic_energy)
    total_energys.append(total_energy)

    # Plot
    plt.figure()
    plt.plot(times, kinetic_energys, label="Kinetic Energy")
    plt.plot(times, electric_energys, label="Electric Energy")
    plt.plot(times, magnetic_energys, label="Magnetic Energy")
    plt.plot(times, total_energys, label="Total Energy")
    plt.title("Energy")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.legend()
    plt.savefig("energy.png")
    plt.close()


class InitializeParticles(Operator):

    @wp.kernel
    def _set_particles(
        particle_position: wp.array2d(dtype=wp.float32),
        particle_velocity: wp.array2d(dtype=wp.float32),
        particle_id: wp.array2d(dtype=wp.uint8),
        lower_bound: wp.vec3f,
        upper_bound: wp.vec3f,
    ):
        # Get particle index
        i = wp.tid()

        # Set particle position
        r = wp.rand_init(i)
        x = lower_bound[0] + (upper_bound[0] - lower_bound[0]) * wp.randf(r)
        y = lower_bound[1] + (upper_bound[1] - lower_bound[1]) * wp.randf(r)
        z = lower_bound[2] + (upper_bound[2] - lower_bound[2]) * wp.randf(r)

        # Set particle velocity
        vx = 0.0
        vy = 0.05 * z
        vz = 0.0

        # Get electron index and proton index
        electron_index = i * 2
        proton_index = i * 2 + 1

        # Set electron info
        particle_position[0, electron_index] = x
        particle_position[1, electron_index] = y
        particle_position[2, electron_index] = z
        particle_velocity[0, electron_index] = vx
        particle_velocity[1, electron_index] = vy
        particle_velocity[2, electron_index] = vz
        particle_id[0, electron_index] = wp.uint8(0)

        # Set proton info
        particle_position[0, proton_index] = x
        particle_position[1, proton_index] = y
        particle_position[2, proton_index] = z
        particle_velocity[0, proton_index] = vx
        particle_velocity[1, proton_index] = -vy
        particle_velocity[2, proton_index] = vz
        particle_id[0, proton_index] = wp.uint8(1)

    def __call__(
        self,
        particle_position,
        particle_velocity,
        particle_id,
        lower_bound,
        upper_bound,
    ):

        wp.launch(
            self._set_particles,
            inputs=[
                particle_position,
                particle_velocity,
                particle_id,
                lower_bound,
                upper_bound,
            ],
            dim=particle_position.shape[1] // 2,
        )

        return particle_position, particle_velocity, particle_id


if __name__ == "__main__":

    # Params
    save_dir = "./particles_in_box"
    dx = 0.10
    origin = (-100.0 * dx, -100.0 * dx, -100.0 * dx)
    length = (200.0 * dx, 200.0 * dx, 200.0 * dx)
    spacing = (dx, dx, dx)
    nr_voxels = (int(length[0]/dx) + 2, int(length[1]/dx) + 2, int(length[2]/dx) + 2) # Add 2 for ghost cells
    print(nr_voxels)
    dt = dx / 4.0
    nr_particles = 3000000
    charge = 10000.0 / nr_particles
    mass = 10000.0 / nr_particles
    #eps = 8.854187817e-12
    #mu = 4.0 * np.pi * 1.0e-7
    #sigma_e = 0.0
    #sigma_m = 0.0
    eps = 1.0
    mu = 1.0
    sigma_e = 0.0
    sigma_m = 0.0
    initial_magnetic_field = np.array([0.0, 0.0, 0.0])

    # Make operators
    update_position = PositionUpdate()
    update_velocity = BorisVelocityUpdate()
    deposit_charge = DepositCharge()
    charge_conservation = ChargeConservation()
    initialize_particles = InitializeParticles()
    magnetic_field_update = MagneticFieldUpdate()
    electric_field_update = ElectricFieldUpdate()

    # Make particle arrays
    particle_position = wp.zeros((3, nr_particles), wp.float32)
    particle_velocity = wp.zeros((3, nr_particles), wp.float32)
    particle_id = wp.zeros((1, nr_particles), wp.uint8)
    particle_mass_mapping = wp.from_numpy(np.array([mass, mass], dtype=np.float32), wp.float32)
    particle_charge_mapping = wp.from_numpy(np.array([charge, -charge], dtype=np.float32), wp.float32)

    # Make electromagnetic arrays
    material_id = wp.zeros(nr_voxels, wp.uint8)
    electric_field = wp.zeros((3, *nr_voxels), wp.float32)
    magnetic_field = wp.from_numpy(np.ones((3, *nr_voxels), dtype=np.float32) * np.expand_dims(initial_magnetic_field, axis=(1, 2, 3)), wp.float32)
    impressed_current = wp.zeros((3, *nr_voxels), wp.float32)
    eps_mapping = wp.from_numpy(np.array([eps], dtype=np.float32), wp.float32)
    mu_mapping = wp.from_numpy(np.array([mu], dtype=np.float32), wp.float32)
    sigma_e_mapping = wp.from_numpy(np.array([sigma_e], dtype=np.float32), wp.float32)
    sigma_m_mapping = wp.from_numpy(np.array([sigma_m], dtype=np.float32), wp.float32)

    # Debugging arrays
    charge_density_0 = wp.zeros((1, *nr_voxels), wp.float32)
    charge_density_1 = wp.zeros((1, *nr_voxels), wp.float32)
    charge_conservation_0 = wp.zeros((1, *nr_voxels), wp.float32)

    # Set initial conditions
    lower_bound = np.array([-10.0 * dx, -10.0 * dx, -10.0 * dx])
    upper_bound = np.array([10.0 * dx, 10.0 * dx, 10.0 * dx])
    particle_position, particle_velocity, particle_id = initialize_particles(
        particle_position,
        particle_velocity,
        particle_id,
        lower_bound,
        upper_bound,
    )

    # Solve loop
    t = 0.0
    plt_freq = 4
    nr_steps = 1024
    import time
    start = time.time()
    for step in tqdm(range(nr_steps)):

        # Zero fields
        impressed_current.zero_()
        charge_density_0.zero_()
        charge_density_1.zero_()
        charge_conservation_0.zero_()

        # Compute charge density
        charge_density_0 = deposit_charge(
            particle_position,
            particle_id,
            charge_density_0,
            particle_charge_mapping,
            origin,
            spacing,
        )

        # Update position
        particle_position = update_position(
            particle_position,
            particle_velocity,
            particle_id,
            impressed_current,
            particle_mass_mapping,
            particle_charge_mapping,
            origin,
            spacing,
            dt,
        )

        # Compute charge density
        charge_density_1 = deposit_charge(
            particle_position,
            particle_id,
            charge_density_1,
            particle_charge_mapping,
            origin,
            spacing,
        )

        # Charge conservation
        charge_conservation_0 = charge_conservation(
            charge_density_0,
            charge_density_1,
            impressed_current,
            charge_conservation_0,
            spacing,
            dt,
        )

        # Update velocity
        particle_velocity = update_velocity(
            particle_position,
            particle_velocity,
            particle_id,
            electric_field,
            magnetic_field,
            particle_mass_mapping,
            particle_charge_mapping,
            origin,
            spacing,
            dt,
        )

        # update fields
        electric_field = electric_field_update(
            electric_field,
            magnetic_field,
            impressed_current,
            material_id,
            eps_mapping,
            sigma_e_mapping,
            spacing=(dx, dx, dx),
            dt=dt,
        )
        magnetic_field = magnetic_field_update(
            magnetic_field,
            electric_field,
            material_id,
            mu_mapping,
            sigma_m_mapping,
            spacing=(dx, dx, dx),
            dt=dt,
        )

        # update time
        t += dt

        # Plot
        if step % plt_freq == 0:
            #plot_solution(
            #    particle_position,
            #    particle_velocity,
            #    impressed_current,
            #    charge_conservation_0,
            #    charge_density_0,
            #    charge_density_1,
            #    save_dir,
            #    step,
            #    origin,
            #    spacing,
            #)
            compute_energy(
                particle_velocity,
                electric_field,
                magnetic_field,
                eps,
                mu,
                mass,
                dx,
                t,
            )
            save_vtk(particle_position, particle_id, charge_density_0, save_dir, step, origin, spacing)

    wp.synchronize()
    end = time.time()
    # compute particle update per second
    print(f"Time per step: {(end - start) / nr_steps}")
    print(f"Million Particle updates per second: {nr_steps * nr_particles / (end - start) / 1000000}")
