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

particle_positions_x = []
particle_positions_y = []
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

    # Add particle position to list
    particle_positions_x.append(particle_position.numpy()[0, 0])
    particle_positions_y.append(particle_position.numpy()[1, 0])
    print(np.max(particle_positions_x), np.min(particle_positions_x))
    print(np.max(particle_positions_y), np.min(particle_positions_y))

    # Plot particle position on x-y plane
    plt.figure()
    plt.scatter(particle_positions_x, particle_positions_y, c="r")
    plt.title(f"Step {step}")
    plt.savefig(f"{save_dir}/particle_position_{str(step).zfill(4)}.png")
    plt.close()

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


def save_vtk(particle_position, save_dir, step):

    # Create point cloud
    grid = pv.PolyData()
    grid.points = particle_position.numpy().T

    # Save
    grid.save(f"{save_dir}/step_{str(step).zfill(4)}.vtk")


if __name__ == "__main__":

    # Params
    save_dir = "./trapped_charges"
    dx = 0.1
    origin = (-5.0, -5.0, -5.0)
    spacing = (dx, dx, dx)
    nr_voxels = (int(10/dx) + 2, int(10/dx) + 2, int(10/dx) + 2) # Add 2 for ghost cells
    dt = dx / 5.0
    nr_particles = 1
    #nr_particles = 1024
    #nr_particles = 256
    charge = 5.0
    mass = 1.0
    magnetic_field = np.array([0.0, 0.0, 1.0])
    electric_field = np.array([0.5, 0.0, 0.0])
    initial_velocity = np.array([0.0, -1.0, 0.0])

    # Make operators
    update_position = PositionUpdate()
    update_velocity = BorisVelocityUpdate()
    deposit_charge = DepositCharge()
    charge_conservation = ChargeConservation()

    # Make arrays
    particle_position = wp.from_numpy(np.array([[1.0], [0.0], [0.0]], dtype=np.float32), wp.float32)
    particle_velocity = wp.from_numpy(np.ones((3, nr_particles), dtype=np.float32) * np.array([initial_velocity]).T, wp.float32)
    particle_id = wp.zeros((1, nr_particles), wp.uint8)
    electric_field = wp.from_numpy(np.ones((3, *nr_voxels), dtype=np.float32) * np.expand_dims(electric_field, axis=(1, 2, 3)), wp.float32)
    magnetic_field = wp.from_numpy(np.ones((3, *nr_voxels), dtype=np.float32) * np.expand_dims(magnetic_field, axis=(1, 2, 3)), wp.float32)
    impressed_current = wp.zeros((3, *nr_voxels), wp.float32)
    charge_density_0 = wp.zeros((1, *nr_voxels), wp.float32)
    charge_density_1 = wp.zeros((1, *nr_voxels), wp.float32)
    charge_conservation_0 = wp.zeros((1, *nr_voxels), wp.float32)
    particle_mass_mapping = wp.from_numpy(np.array([mass], dtype=np.float32), wp.float32)
    particle_charge_mapping = wp.from_numpy(np.array([charge], dtype=np.float32), wp.float32)

    # Solve loop
    nr_steps = 1024
    t = 0.0
    plt_freq = 1
    import time
    start = time.time()
    for step in tqdm(range(nr_steps)):

        # Zero fields
        impressed_current.zero_()
        charge_density_0.zero_()
        charge_density_1.zero_()
        charge_conservation_0.zero_()

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

        # update time
        t += dt

        # Plot
        if step % plt_freq == 0:
            plot_solution(
                particle_position,
                particle_velocity,
                impressed_current,
                charge_conservation_0,
                charge_density_0,
                charge_density_1,
                save_dir,
                step,
                origin,
                spacing,
            )
            #save_vtk(particle_position, save_dir, step)

    wp.synchronize()
    end = time.time()
    # compute particle update per second
    print(f"Time per step: {(end - start) / nr_steps}")
    print(f"Million Particle updates per second: {nr_steps * nr_particles / (end - start) / 1000000}")
