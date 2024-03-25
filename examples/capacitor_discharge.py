# Simple example of voxelizing a build123d geometry

import warp as wp
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyvista as pv

wp.init()

from dense_plasma_focus.geometry.circuit.capacitor_discharge import CapacitorDischarge
from dense_plasma_focus.operator.operator import Operator
from dense_plasma_focus.operator.electromagnetism.electromagnetism import ElectricFieldUpdate, MagneticFieldUpdate, ElectricFieldToChargeDensity
from dense_plasma_focus.operator.electromagnetism.boundary_conditions import EMSetBoundary

from dense_plasma_focus.operator.voxelize.build123d import Build123D, get_materials_in_compound

def plot_solution(electric_field, magnetic_field, material_id, save_dir, t, step, tau, tau_switch_start):

    # Plot voxels, electric field and magnetic field
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    # Get y plane
    y_slice = material_id.shape[1] // 2
    material_id_slice = material_id.numpy()[..., y_slice, :]
    electric_field_slice = np.linalg.norm(electric_field.numpy()[..., y_slice, :], axis=0)
    magnetic_field_slice = np.linalg.norm(magnetic_field.numpy()[..., y_slice, :], axis=0)

    # material_id plot
    im1 = axes[0].imshow(material_id_slice, cmap='inferno')
    axes[0].set_title('Material ID')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # electric_field plot
    im2 = axes[1].imshow(electric_field_slice, cmap='inferno')
    axes[1].set_title('Electric Field')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # magnetic_field plot
    im3 = axes[2].imshow(magnetic_field_slice, cmap='inferno')
    axes[2].set_title('Magnetic Field')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # Set title
    if t < tau:
        plt.suptitle(f'Charging capacitor at t (us) = {t / 1e-6:.3f}', fontsize=32)
    elif t < tau_switch_start:
        plt.suptitle(f'Steady State at t (us) = {t / 1e-6:.4f}', fontsize=32)
    else:
        plt.suptitle(f'Discharging capacitor at t (us) = {t / 1e-6:.3f}', fontsize=32)

    # Layout adjustment and saving the figure
    save_dir = "./capacitor_discharge"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/capacitor_discharge_{str(step).zfill(6)}.png")
    plt.close(fig)

current_time = []
total_charge_right = []
total_charge_left = []
def plot_charge_density(charge_density, material_id, save_dir, t, step):

    # Get total charge
    z_slice = 53
    np_charge_density = charge_density.numpy()
    total_charge_left.append(np.sum(np_charge_density[..., :z_slice]))
    total_charge_right.append(np.sum(np_charge_density[..., z_slice:]))

    # Get time
    current_time.append(t)

    # Plot charge density over time
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(current_time, total_charge_left, label="Left")
    ax.plot(current_time, total_charge_right, label="Right")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Total Charge (C)')
    ax.set_title('Total Charge over Time')
    plt.savefig(f"{save_dir}/total_charge.png")
    plt.close(fig)




def save_vtk(electric_field, magnetic_field, material_id, save_dir, step, origin, spacing):

    # Create grid
    grid = pv.ImageData()
    #grid.dimensions = [s + 1 for s in material_id.shape]
    grid.dimensions = [s for s in material_id.shape]
    grid.origin = origin
    grid.spacing = spacing

    # Get data
    np_material_id = material_id.numpy().flatten('F')
    np_electric_field = electric_field.numpy().transpose(1, 2, 3, 0).reshape(-1, 3, order='F')
    np_magnetic_field = magnetic_field.numpy().transpose(1, 2, 3, 0).reshape(-1, 3, order='F')

    # Add data
    grid.point_data["Material_ID"] = np_material_id
    grid.point_data["Electric_Field"] = np_electric_field
    grid.point_data["Magnetic_Field"] = np_magnetic_field

    # Save grid
    grid.save(f"{save_dir}/capacitor_discharge_{str(step).zfill(6)}.vtk")
    

class SetCapacitorElectricFeild(Operator):

    @wp.kernel
    def _set_electric_field(
        electric_field: wp.array4d(dtype=wp.float32),
        material_id: wp.array3d(dtype=wp.uint8),
        dielectric_material_id: wp.uint8,
        field_strength: wp.float32,
    ):
        # Get index
        i, j, k = wp.tid()
        i += 1
        j += 1
        k += 1

        # Get material id
        mat_id = material_id[i, j, k]

        # Set electric field if material is dielectric (include edges)
        if mat_id == dielectric_material_id:
            electric_field[2, i, j, k] = field_strength
            electric_field[2, i+1, j, k] = field_strength
            electric_field[2, i, j+1, k] = field_strength
            electric_field[2, i+1, j+1, k] = field_strength

    def __call__(
        self,
        electric_field: wp.array4d(dtype=wp.float32),
        material_id: wp.array3d(dtype=wp.uint8),
        dielectric_material_id: wp.uint8,
        field_strength: wp.float32,
    ):

        wp.launch(
            self._set_electric_field,
            inputs=[
                electric_field,
                material_id,
                dielectric_material_id,
                field_strength,
            ],
            dim=[s-2 for s in electric_field.shape],
        )

        return electric_field

if __name__ == "__main__":

    # Params
    save_dir = "./capacitor_discharge"
    dx = 1.0
    origin = (-50.0, -50.0, -50.0)
    nr_voxels = (int(100/dx), int(100/dx), int(100/dx))
    c = 299792458.0
    dt = dx / (c * np.sqrt(3.0))
    tau = 1.0e-6 # time to charge capacitor
    tau_switch_start = 1.5e-6 # time to switch on switch
    tau_switch_end = 2.0e-6 # time that switch is fully on
    tau_end = 2000.0e-6 # time to end simulation
    e0 = 1.0 # Final electric field strength in capacitor's dielectric
    anode_resistance = 1.0e-10 # resistance of switch when fully on
    cathode_resistance = 1.0e-10 # constant through simulation
    capacitance = 1.0e-6 # F
    capacitor_dielectric_name = "dielectric"
    anode_resistor_name = "anode_resistor"
    cathode_resistor_name = "cathode_resistor"

    # Create geometry
    geometry = CapacitorDischarge(
        capacitor_dielectric_name=capacitor_dielectric_name,
        capacitance=capacitance,
        anode_resistance=anode_resistance,
        cathode_resistance=cathode_resistance,
        anode_resistor_name=anode_resistor_name,
        cathode_resistor_name=cathode_resistor_name,
    )

    # Get materials in geometry
    materials = get_materials_in_compound(geometry)
    dielectric_material_id = [i for i, m in enumerate(materials) if m.name == capacitor_dielectric_name][0]
    anode_resistor_id = [i for i, m in enumerate(materials) if m.name == anode_resistor_name][0]
    cathode_resistor_id = [i for i, m in enumerate(materials) if m.name == cathode_resistor_name][0]

    # Make operators
    voxelizer = Build123D()
    electric_field_update = ElectricFieldUpdate()
    magnetic_field_update = MagneticFieldUpdate()
    electric_field_to_charge_density = ElectricFieldToChargeDensity()
    boundary_conditions = EMSetBoundary()
    set_capacitor_electric_field = SetCapacitorElectricFeild()

    # Make arrays
    material_id = wp.zeros(nr_voxels, wp.uint8)
    electric_field = wp.zeros((3, *nr_voxels), wp.float32)
    magnetic_field = wp.zeros((3, *nr_voxels), wp.float32)
    charge_density = wp.zeros((1, *nr_voxels), wp.float32)
    impressed_current = wp.zeros((3, *nr_voxels), wp.float32)
    eps_mapping = wp.from_numpy(np.array([m.eps for m in materials], dtype=np.float32))
    mu_mapping = wp.from_numpy(np.array([m.mu for m in materials], dtype=np.float32))
    sigma_e_mapping = wp.from_numpy(np.array([m.sigma_e for m in materials], dtype=np.float32))
    sigma_m_mapping = wp.from_numpy(np.array([m.sigma_m for m in materials], dtype=np.float32))

    # Get finale conductivity of anode resistor
    final_sigma_e = [m.sigma_e for m in materials][anode_resistor_id]
    print(f"Final sigma_e: {final_sigma_e}")

    # Set conductivity to zero in beginning
    sigma_e_mapping = sigma_e_mapping.numpy() # TODO Hacky
    sigma_e_mapping[anode_resistor_id] = 0.0
    sigma_e_mapping = wp.from_numpy(sigma_e_mapping)

    # Voxelize electrode
    material_id = voxelizer(
        voxels=material_id,
        compound=geometry,
        spacing=(dx, dx, dx),
        origin=origin,
        shape=nr_voxels,
        materials=materials,
        nr_processes=32,
    )

    # Solve loop
    t = 0.0
    nr_steps = int(tau_end / dt)
    plt_freq = 30
    for step in tqdm(range(nr_steps)):

        # if t < tau then charge capacitor
        if t < tau:
            alpha = (4.0 / tau)**2
            e_x = e0 * np.exp(-alpha * (t - tau)**2)
            electric_field = set_capacitor_electric_field(
                electric_field,
                material_id,
                dielectric_material_id,
                field_strength=e_x,
            )
        # if tau_switch_start < t < tau_switch_end then switches sigma_e will be increased
        elif (t >= tau_switch_start) and (t < tau_switch_end):
            alpha = (4.0 / (tau_switch_end - tau_switch_start))**2
            new_sigma_e = final_sigma_e * np.exp(-alpha * (t - tau_switch_end)**2)
            sigma_e_mapping = sigma_e_mapping.numpy()
            sigma_e_mapping[anode_resistor_id] = new_sigma_e
            sigma_e_mapping = wp.from_numpy(sigma_e_mapping)
            print(f"New sigma_e: {new_sigma_e}")

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

        # boundary conditions
        electric_field, magnetic_field = boundary_conditions(
            electric_field,
            magnetic_field,
            spacing=(dx, dx, dx),
            dt=dt,
        )

        # Convert electric field to charge density
        charge_density = electric_field_to_charge_density(
            electric_field,
            charge_density,
            material_id,
            sigma_e_mapping,
            spacing=(dx, dx, dx),
        )

        # update time
        t += dt

        # Plot
        if step % plt_freq == 0:
            #plot_solution(electric_field, magnetic_field, material_id, save_dir, t, step, tau, tau_switch_start)
            plot_charge_density(charge_density, material_id, save_dir, t, step)
            #save_vtk(electric_field, magnetic_field, material_id, save_dir, step, origin, (dx, dx, dx))
