# FDTD solver for 3D Yee grid

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from dataclasses import dataclass
from tqdm import tqdm


import warp as wp

wp.init()

@dataclass
class Bounds:
    x_lower: str
    x_upper: str
    y_lower: str
    y_upper: str
    z_lower: str
    z_upper: str


def construct_material_sampler(
    materials,
    bounds,
):

    # Get list of material probabilities
    eps = tuple([material.eps for material in materials])
    mu = tuple([material.mu for material in materials])
    sigma_e = tuple([material.sigma_e for material in materials])
    sigma_m = tuple([material.sigma_m for material in materials])

    @wp.func
    def sample_material(
        material: wp.array3d,
        i: wp.int32,
        j: wp.int32,
        k: wp.int32,
        max_i: wp.int32,
        max_j: wp.int32,
        max_k: wp.int32,
    ):

        # Get material id for needed cells
        m_1_1_1 = material[i, j, k]
        m_0_1_1 = material[i - 1, j, k]
        m_1_0_1 = material[i, j - 1, k]
        m_1_1_0 = material[i, j, k - 1]

        # Get eps
        eps_x = (eps[m_1_1_1] + eps[m_0_1_1] + eps[m_1_0_1] + eps[m_1_1_0]) / 4.0

        return eps

    return sample_material




def set_box(
        material, 
        eps,
        mu,
        sigma_e,
        sigma_m,
        color,
        origin,
        size):
    """
    Set a box of material in the grid.
    """
    # Get box indices
    x0, y0, z0 = origin
    x1, y1, z1 = x0 + size[0], y0 + size[1], z0 + size[2]

    # Set material properties
    eps = eps.at[x0:x1, y0:y1, z0:z1].set(material.eps)
    mu = mu.at[x0:x1, y0:y1, z0:z1].set(material.mu)
    sigma_e = sigma_e.at[x0:x1, y0:y1, z0:z1].set(material.sigma_e)
    sigma_m = sigma_m.at[x0:x1, y0:y1, z0:z1].set(material.sigma_m)
 
    color = color.at[x0:x1, y0:y1, z0:z1].set(material.color)

    return eps, mu, sigma_e, sigma_m, color

def get_material_properties_xyz(
        eps,
        mu,
        sigma_e,
        sigma_m):

    # Get material properties on staggered grid
    # eps
    eps_x = (eps + jnp.roll(eps, 1, axis=1) + jnp.roll(eps, 1, axis=2) + jnp.roll(jnp.roll(eps, 1, axis=1), 1, axis=2)) / 4.0
    eps_y = (eps + jnp.roll(eps, 1, axis=0) + jnp.roll(eps, 1, axis=2) + jnp.roll(jnp.roll(eps, 1, axis=0), 1, axis=2)) / 4.0
    eps_z = (eps + jnp.roll(eps, 1, axis=0) + jnp.roll(eps, 1, axis=1) + jnp.roll(jnp.roll(eps, 1, axis=0), 1, axis=1)) / 4.0

    # mu
    mu_x = (2.0 * mu * jnp.roll(mu, 1, axis=0)) / (mu + jnp.roll(mu, 1, axis=0) + 1e-10)
    mu_y = (2.0 * mu * jnp.roll(mu, 1, axis=1)) / (mu + jnp.roll(mu, 1, axis=1) + 1e-10)
    mu_z = (2.0 * mu * jnp.roll(mu, 1, axis=2)) / (mu + jnp.roll(mu, 1, axis=2) + 1e-10)

    # sigma_e
    sigma_e_x = (sigma_e + jnp.roll(sigma_e, 1, axis=1) + jnp.roll(sigma_e, 1, axis=2) + jnp.roll(jnp.roll(sigma_e, 1, axis=1), 1, axis=2)) / 4.0
    sigma_e_y = (sigma_e + jnp.roll(sigma_e, 1, axis=0) + jnp.roll(sigma_e, 1, axis=2) + jnp.roll(jnp.roll(sigma_e, 1, axis=0), 1, axis=2)) / 4.0
    sigma_e_z = (sigma_e + jnp.roll(sigma_e, 1, axis=0) + jnp.roll(sigma_e, 1, axis=1) + jnp.roll(jnp.roll(sigma_e, 1, axis=0), 1, axis=1)) / 4.0

    # sigma_m
    sigma_m_x = (2.0 * sigma_m * jnp.roll(sigma_m, 1, axis=0)) / (sigma_m + jnp.roll(sigma_m, 1, axis=0) + 1e-10)
    sigma_m_y = (2.0 * sigma_m * jnp.roll(sigma_m, 1, axis=1)) / (sigma_m + jnp.roll(sigma_m, 1, axis=1) + 1e-10)
    sigma_m_z = (2.0 * sigma_m * jnp.roll(sigma_m, 1, axis=2)) / (sigma_m + jnp.roll(sigma_m, 1, axis=2) + 1e-10)

    # Return material properties
    eps_xyz = jnp.stack([eps_x, eps_y, eps_z], axis=-1)
    mu_xyz = jnp.stack([mu_x, mu_y, mu_z], axis=-1)
    sigma_e_xyz = jnp.stack([sigma_e_x, sigma_e_y, sigma_e_z], axis=-1)
    sigma_m_xyz = jnp.stack([sigma_m_x, sigma_m_y, sigma_m_z], axis=-1)

    return eps_xyz, mu_xyz, sigma_e_xyz, sigma_m_xyz

#@partial(jax.jit, static_argnums=(3,4))
def fix_e_y(
        electric_field,
        e0,
        time,
        tau,
        origin,
        size):
    """
    Fix the x component of the electric field in box.
    """
    # Get box indices
    x0, y0, z0 = origin
    x1, y1, z1 = x0 + size[0], y0 + size[1], z0 + size[2]

    # Fix electric field
    alpha = (4.0 / tau) ** 2
    electric_field = electric_field.at[x0:x1+1, y0:y1, z0:z1+1, 1].set(e0 * jnp.exp(-alpha * (time - tau) ** 2))

    return electric_field

def update_switch_conductivity(
        sigma_e,
        time,
        tau_start,
        tau_end,
        sigma_switch,
        origin,
        size):
    """
    Update the conductivity of the switch.
    """
    # Get box indices
    x0, y0, z0 = origin
    x1, y1, z1 = x0 + size[0], y0 + size[1], z0 + size[2]

    # Update conductivity
    alpha = (4.0 / (tau_end - tau_start)) ** 2
    sigma_e = sigma_e.at[x0:x1, y0:y1, z0:z1].set(sigma_switch * jnp.exp(-alpha * (time - tau_end) ** 2))

    return sigma_e

@partial(jax.jit, donate_argnums=(0))
def update_electric_field(
        electric_field,
        magnetic_field,
        impressed_electric_current,
        eps,
        sigma_e,
        dt,
        dx,
):

    # Get x coefficients
    c_ee = (2 * eps - sigma_e * dt) / (2 * eps + sigma_e * dt)
    c_eh = 2 * dt / (dx * (2 * eps + sigma_e * dt))
    c_ej = - 2 * dt / (2 * eps + sigma_e * dt)

    # Get rolled magnetic field
    h_1_1_1 = magnetic_field
    h_0_1_1 = jnp.roll(h_1_1_1, 1, axis=0)
    h_1_0_1 = jnp.roll(h_1_1_1, 1, axis=1)
    h_1_1_0 = jnp.roll(h_1_1_1, 1, axis=2)

    # Apply nueumann boundary conditions
    h_0_1_1 = h_0_1_1.at[0, :, :].set(0.0)
    h_1_0_1 = h_1_0_1.at[:, 0, :].set(0.0)
    h_1_1_0 = h_1_1_0.at[:, :, 0].set(0.0)

    # Update electric field
    new_electric_field_x = (
        c_ee[..., 0:1] * electric_field[..., 0:1]
        + c_eh[..., 0:1] * (
            + (h_1_1_1[..., 2:3] - h_1_0_1[..., 2:3])
            - (h_1_1_1[..., 1:2] - h_1_1_0[..., 1:2])
        )
        + c_ej[..., 0:1] * impressed_electric_current[..., 0:1]
    )
    new_electric_field_y = (
        c_ee[..., 1:2] * electric_field[..., 1:2]
        + c_eh[..., 1:2] * (
            + (h_1_1_1[..., 0:1] - h_1_1_0[..., 0:1])
            - (h_1_1_1[..., 2:3] - h_0_1_1[..., 2:3])
        )
        + c_ej[..., 1:2] * impressed_electric_current[..., 1:2]
    )
    new_electric_field_z = (
        c_ee[..., 2:3] * electric_field[..., 2:3]
        + c_eh[..., 2:3] * (
            + (h_1_1_1[..., 1:2] - h_0_1_1[..., 1:2])
            - (h_1_1_1[..., 0:1] - h_1_0_1[..., 0:1])
        )
        + c_ej[..., 2:3] * impressed_electric_current[..., 2:3]
    )

    return jnp.concatenate([new_electric_field_x, new_electric_field_y, new_electric_field_z], axis=-1)


@partial(jax.jit, donate_argnums=(1))
def update_magnetic_field(
        electric_field,
        magnetic_field,
        impressed_magnetic_current,
        mu,
        sigma_m,
        dt,
        dx
):

    # Get coefficients
    c_hh = (2 * mu - sigma_m * dt) / (2 * mu + sigma_m * dt)
    c_he = 2 * dt / (dx * (2 * mu + sigma_m * dt))
    c_hm = - 2 * dt / (2 * mu + sigma_m * dt)

    # Get rolled electric field
    e_0_0_0 = electric_field
    e_1_0_0 = jnp.roll(e_0_0_0, -1, axis=0)
    e_0_1_0 = jnp.roll(e_0_0_0, -1, axis=1)
    e_0_0_1 = jnp.roll(e_0_0_0, -1, axis=2)

    # Apply nueumann boundary conditions
    e_1_0_0 = e_1_0_0.at[-1, :, :].set(0.0)
    e_0_1_0 = e_0_1_0.at[:, -1, :].set(0.0)
    e_0_0_1 = e_0_0_1.at[:, :, -1].set(0.0)

    # Update magnetic field
    new_magnetic_field_x = (
        c_hh[..., 0:1] * magnetic_field[..., 0:1]
        + c_he[..., 0:1] * (
            + (e_0_0_1[..., 1:2] - e_0_0_0[..., 1:2])
            - (e_0_1_0[..., 2:3] - e_0_0_0[..., 2:3])
        )
        + c_hm[..., 0:1] * impressed_magnetic_current[..., 0:1]
    )
    new_magnetic_field_y = (
        c_hh[..., 1:2] * magnetic_field[..., 1:2]
        + c_he[..., 1:2] * (
            + (e_1_0_0[..., 2:3] - e_0_0_0[..., 2:3])
            - (e_0_0_1[..., 0:1] - e_0_0_0[..., 0:1])
        )
        + c_hm[..., 1:2] * impressed_magnetic_current[..., 1:2]
    )
    new_magnetic_field_z = (
        c_hh[..., 2:3] * magnetic_field[..., 2:3]
        + c_he[..., 2:3] * (
            + (e_0_1_0[..., 0:1] - e_0_0_0[..., 0:1])
            - (e_1_0_0[..., 1:2] - e_0_0_0[..., 1:2])
        )
        + c_hm[..., 2:3] * impressed_magnetic_current[..., 2:3]
    )

    return jnp.concatenate([new_magnetic_field_x, new_magnetic_field_y, new_magnetic_field_z], axis=-1)

@partial(jax.jit)
def charge_density(
        electric_field,
        eps,
):
    """
    Calculate the charge density.
    """
    # Get divergence of electric field
    e_0_0_0 = electric_field
    e_1_0_0 = jnp.roll(e_0_0_0, -1, axis=0)
    e_0_1_0 = jnp.roll(e_0_0_0, -1, axis=1)
    e_0_0_1 = jnp.roll(e_0_0_0, -1, axis=2)
    div_e = (
        (e_1_0_0[..., 0:1] - e_0_0_0[..., 0:1])
        + (e_0_1_0[..., 1:2] - e_0_0_0[..., 1:2])
        + (e_0_0_1[..., 2:3] - e_0_0_0[..., 2:3])
    )

    # Get charge density
    rho = div_e / eps[..., 0:1]

    return rho



if __name__ == '__main__':

    # Initialize parameters
    c = 299792458.0  # Speed of light
    dx = 2e-3  # spatial step
    size = 64  # Size of the simulation grid
    dt = dx / (c * np.sqrt(3.0))  # Time step
    solve_time = 2e-6  # Time to solve for
    nr_steps = int(solve_time // dt)  # Number of time steps
    tau = 0.25e-6  # Time to charge the capacitor
    tau_switch_start = 0.5e-6
    tau_switch_end = 0.75e-6
    scale_factor = 2
    add_origin = (scale_factor * size // 2) - (size // 2)

    # Initialize material properties
    eps = jnp.ones((size*scale_factor, size*scale_factor, size*scale_factor)) * Vacuum.eps
    mu = jnp.ones((size*scale_factor, size*scale_factor, size*scale_factor)) * Vacuum.mu
    sigma_e = jnp.ones((size*scale_factor, size*scale_factor, size*scale_factor)) * Vacuum.sigma_e
    sigma_m = jnp.ones((size*scale_factor, size*scale_factor, size*scale_factor)) * Vacuum.sigma_m
    color = jnp.zeros((size*scale_factor, size*scale_factor, size*scale_factor)) * Vacuum.color

    # Add copper 
    eps, mu, sigma_e, sigma_m, color = set_box(
        material=Copper,
        eps=eps,
        mu=mu,
        sigma_e=sigma_e,
        sigma_m=sigma_m,
        color=color,
        origin=(size // 4 + add_origin, size // 4 + add_origin, size // 4 + add_origin),
        size=(size // 2, size // 16, size // 2),
    )
    eps, mu, sigma_e, sigma_m, color = set_box(
        material=Copper,
        eps=eps,
        mu=mu,
        sigma_e=sigma_e,
        sigma_m=sigma_m,
        color=color,
        origin=(size // 4 + add_origin, 3 * size // 4 + add_origin, size // 4 + add_origin),
        size=(size // 2, size // 16, size // 2),
    )
    eps, mu, sigma_e, sigma_m, color = set_box(
        material=Quartz,
        eps=eps,
        mu=mu,
        sigma_e=sigma_e,
        sigma_m=sigma_m,
        color=color,
        origin=(size // 4 + add_origin, size // 4 + size // 16 + add_origin, size // 4 + add_origin),
        size=(size // 8, size // 2 - size // 16, size // 2),
    )
    eps, mu, sigma_e, sigma_m, color = set_box(
       material=Switch,
       eps=eps,
       mu=mu,
       sigma_e=sigma_e,
       sigma_m=sigma_m,
       color=color,
       origin=(3 * size // 4 - size // 8 + add_origin, size // 4 + size // 16 + add_origin, size // 4 + add_origin),
       size=(size // 8, size // 2 - size // 16, size // 2),
    )
    eps, mu, sigma_e, sigma_m, color = set_box(
         material=Absorber,
         eps=eps,
         mu=mu,
         sigma_e=sigma_e,
         sigma_m=sigma_m,
         color=color,
         origin=(0, 0, 0),
         size=(size*scale_factor, size//16, size*scale_factor),
    )
    eps, mu, sigma_e, sigma_m, color = set_box(
            material=Absorber,
            eps=eps,
            mu=mu,
            sigma_e=sigma_e,
            sigma_m=sigma_m,
            color=color,
            origin=(0, 0, 0),
            size=(size*scale_factor, size*scale_factor, size//16),
    )
    eps, mu, sigma_e, sigma_m, color = set_box(
            material=Absorber,
            eps=eps,
            mu=mu,
            sigma_e=sigma_e,
            sigma_m=sigma_m,
            color=color,
            origin=(0, 0, 0),
            size=(size//16, size*scale_factor, size*scale_factor),
    )
    eps, mu, sigma_e, sigma_m, color = set_box(
            material=Absorber,
            eps=eps,
            mu=mu,
            sigma_e=sigma_e,
            sigma_m=sigma_m,
            color=color,
            origin=(size*scale_factor - size//16, 0, 0),
            size=(size//16, size*scale_factor, size*scale_factor),
    )
    eps, mu, sigma_e, sigma_m, color = set_box(
            material=Absorber,
            eps=eps,
            mu=mu,
            sigma_e=sigma_e,
            sigma_m=sigma_m,
            color=color,
            origin=(0, size*scale_factor - size//16, 0),
            size=(size*scale_factor, size//16, size*scale_factor),
    )
    eps, mu, sigma_e, sigma_m, color = set_box(
            material=Absorber,
            eps=eps,
            mu=mu,
            sigma_e=sigma_e,
            sigma_m=sigma_m,
            color=color,
            origin=(0, 0, size*scale_factor - size//16),
            size=(size*scale_factor, size*scale_factor, size//16),
    )


    # Get material properties on staggered grid
    eps_xyz, mu_xyz, sigma_e_xyz, sigma_m_xyz = get_material_properties_xyz(
            eps,
            mu,
            sigma_e,
            sigma_m,
    )

    # Initialize fields
    electric_field = jnp.zeros((size*scale_factor, size*scale_factor, size*scale_factor, 3))
    magnetic_field = jnp.zeros((size*scale_factor, size*scale_factor, size*scale_factor, 3))

    # Initialize impressed current
    impressed_electric_current = jnp.zeros((size*scale_factor, size*scale_factor, size*scale_factor, 3))
    impressed_magnetic_current = jnp.zeros((size*scale_factor, size*scale_factor, size*scale_factor, 3))
    
    # FDTD simulation loop
    time = 0.0
    plot_freq = int(1e-9 // dt)
    for t in tqdm(range(nr_steps)):
        # Update electric field
        electric_field = update_electric_field(
                electric_field,
                magnetic_field,
                impressed_electric_current,
                eps_xyz,
                sigma_e_xyz,
                dt,
                dx,
        )

        # Fix electric field in quartz
        if time < tau:
            electric_field = fix_e_y(
                    electric_field,
                    e0=Copper.eps,
                    time=time,
                    tau=tau,
                    origin=(size // 4 + add_origin, size // 4 + size // 16 + add_origin, size // 4 + add_origin),
                    size=(size // 8, size // 2 - size // 16, size // 2),
            )

        # Flip circuit (wait a bit for the wave to propagate)
        if time > tau_switch_start:
            sigma_e = update_switch_conductivity(
                    sigma_e,
                    time=time,
                    tau_start=tau_switch_start,
                    tau_end=tau_switch_end,
                    sigma_switch=1.0,
                    origin=(3 * size // 4 - size // 8 + add_origin, size // 4 + size // 16 + add_origin, size // 4 + add_origin),
                    size=(size // 8, size // 2 - size // 16, size // 2),
            )
            eps_xyz, mu_xyz, sigma_e_xyz, sigma_m_xyz = get_material_properties_xyz(
                    eps,
                    mu,
                    sigma_e,
                    sigma_m,
            )

        # Update magnetic field
        magnetic_field = update_magnetic_field(
                electric_field,
                magnetic_field,
                impressed_magnetic_current,
                mu_xyz,
                sigma_m_xyz,
                dt,
                dx,
        )

        # Update time
        time += dt
    
        # Visualization at certain intervals
        if t % plot_freq == 0:

            fig, axes = plt.subplots(2, 4, figsize=(20, 10))

            # Electric field plot (z plane)
            im1 = axes[0, 0].imshow(electric_field[:, :, scale_factor * size // 2, 1], cmap='viridis')
            axes[0, 0].set_title('Electric field Y, z plane')
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

            # Electric field plot (y plane)
            im1 = axes[0, 1].imshow(electric_field[:, scale_factor * size // 2, :, 1], cmap='viridis')
            axes[0, 1].set_title('Electric field Y, y plane')
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

            # Magnetic field plot (z plane)
            im2 = axes[0, 2].imshow(magnetic_field[:, :, scale_factor * size // 2, 2], cmap='plasma')
            axes[0, 2].set_title('Magnetic field Z, z plane')
            plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

            # Magnetic field plot (y plane)
            im2 = axes[0, 3].imshow(magnetic_field[:, scale_factor * size // 2, :, 2], cmap='plasma')
            axes[0, 3].set_title('Magnetic field Z, y plane')
            plt.colorbar(im2, ax=axes[0, 3], fraction=0.046, pad=0.04)

            # Charge density plot (z plane)
            rho = charge_density(electric_field, eps)
            im4 = axes[1, 0].imshow(rho[:, :, scale_factor * size // 2], cmap='RdBu')
            axes[1, 0].set_title('Charge density, z plane')
            plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

            # Charge density plot (y plane)
            im4 = axes[1, 1].imshow(rho[:, scale_factor * size // 2, :], cmap='RdBu')
            axes[1, 1].set_title('Charge density, y plane')
            plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

            # Color plot
            im3 = axes[1, 2].imshow(color[:, :, scale_factor * size // 2], cmap='inferno')
            axes[1, 2].set_title('Material Color, z plane')
            plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

            # Color plot
            im3 = axes[1, 3].imshow(color[:, scale_factor * size // 2, :], cmap='inferno')
            axes[1, 3].set_title('Material Color, y plane')
            plt.colorbar(im3, ax=axes[1, 3], fraction=0.046, pad=0.04)

            # Title
            if time < tau:
                plt.suptitle(f'Charging capacitor at t (us) = {time / 1e-6:.3f}', fontsize=32)
            elif time < tau_switch_start:
                plt.suptitle(f'Steady State at t (us) = {time / 1e-6:.4f}', fontsize=32)
            else:
                plt.suptitle(f'Discharging capacitor at t (us) = {time / 1e-6:.3f}', fontsize=32)

            # Layout adjustment and saving the figure
            plt.savefig(f'./images/{str(t//plot_freq).zfill(10)}.png')
            plt.close(fig)
