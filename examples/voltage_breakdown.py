# Simple example of voxelizing a build123d geometry

import warp as wp
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyvista as pv

wp.init()

from dense_plasma_focus.geometry.circuit.breakdown_circuit import BreakdownCircuit
from dense_plasma_focus.particle import Particle, Electron, Proton
from dense_plasma_focus.operator.operator import Operator
from dense_plasma_focus.operator.electromagnetism.electromagnetism import ElectricFieldUpdate, MagneticFieldUpdate
from dense_plasma_focus.operator.electromagnetism.boundary_conditions import EMSetBoundary
from dense_plasma_focus.operator.pic.particle_pusher import BorisVelocityUpdate, PositionUpdate, DepositCharge, ChargeConservation
from dense_plasma_focus.operator.voxelize.build123d import Build123DVoxelize, get_materials_in_compound

def save_vtk(
    particle_position,
    particle_velocity,
    particle_id,
    electric_field,
    magnetic_field,
    current_density,
    material_id,
    save_dir,
    step,
    origin,
    spacing
):

    # Create point cloud
    grid = pv.PolyData(particle_position.numpy().T)
    grid["id"] = particle_id.numpy().T
    grid["velocity"] = particle_velocity.numpy().T
    grid.save(f"{save_dir}/particles_{str(step).zfill(4)}.vtk")

    # Create grid
    grid = pv.ImageData()
    grid.dimensions = [material_id.shape[2], material_id.shape[1], material_id.shape[0]]
    grid.origin = origin
    grid.spacing = spacing

    # Get data
    #np_charge_density = charge_density.numpy().flatten('F')
    np_material_id = material_id.numpy().flatten('F')
    np_current_density = current_density.numpy().transpose(1, 2, 3, 0).reshape(-1, 3, order='F')
    np_electric_field = electric_field.numpy().transpose(1, 2, 3, 0).reshape(-1, 3, order='F')
    np_magnetic_field = magnetic_field.numpy().transpose(1, 2, 3, 0).reshape(-1, 3, order='F')

    # Add data
    #grid.point_data["charge_density"] = np_charge_density
    grid.point_data["material_id"] = np_material_id
    grid.point_data["current_density"] = np_current_density
    grid.point_data["electric_field"] = np_electric_field
    grid.point_data["magnetic_field"] = np_magnetic_field

    # Save grid
    grid.save(f"{save_dir}/volume_{str(step).zfill(4)}.vtk")


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
        vy = 0.0
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
        particle_id[0, electron_index] = wp.uint8(1)

        # Set proton info
        particle_position[0, proton_index] = x
        particle_position[1, proton_index] = y
        particle_position[2, proton_index] = z
        particle_velocity[0, proton_index] = vx
        particle_velocity[1, proton_index] = vy
        particle_velocity[2, proton_index] = vz
        particle_id[0, proton_index] = wp.uint8(2)

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

    # Domain Params
    save_dir = "./voltage_breakdown"
    dx = 0.0002 # m
    #dx = 0.0005 # m
    c = 299792458.0
    origin = (-0.05, -0.025, -0.025)
    length = (0.075, 0.05, 0.075)
    spacing = (dx, dx, dx)
    nr_voxels = (int(length[0]/dx) + 2, int(length[1]/dx) + 2, int(length[2]/dx) + 2) # Add 2 for ghost cells
    print(f"Nr voxels: {nr_voxels}")
    dt = dx / (c * np.sqrt(3))

    # Capacitor Params
    voltage = 1e3 # V
    plate_separation = 0.01 # m
    electric_field_strength = voltage / plate_separation
    tau = 10.0e-9 # time to charge capacitor
    tau_switch_start = 20.0e-9 # time to switch on switch
    tau_switch_end = 50.0e-9 # time that switch is fully on
    tau_end = 1000.0e-9 # time to end simulation
    capacitance = 1.0e-9 # F
    cathode_resistance = 1.0e-6 # constant through simulation
    anode_resistance = 1.0e-6 # resistance of switch when fully on
    capacitor_dielectric_name = "dielectric"
    anode_resistor_name = "anode_resistor"
    cathode_resistor_name = "cathode_resistor"
    total_charge = capacitance * voltage
    print(f"Total charge: {total_charge}")

    total_electrons = total_charge / 1.6e-19
    print(f"Total electrons: {total_electrons}")

    # Particle Params
    #number_density = 1e26 # m^-3
    particle_box = 0.005 # m
    nr_particles = int(2e6)
    #nr_particles = int(1e5)
    micro_to_macro_ratio = 1.0e8 * total_electrons / nr_particles
    electron = Electron(micro_to_macro_ratio)
    proton = Proton(micro_to_macro_ratio)
    print(f"Micro to macro ratio: {micro_to_macro_ratio}")
    print(f"Electron mass: {electron.mass}")
    print(f"Electron charge: {electron.charge}")
    print(f"Proton mass: {proton.mass}")
    print(f"Proton charge: {proton.charge}")

    # Print total joules in capacitor
    joules = 0.5 * capacitance * voltage**2
    print(f"Joules in capacitor: {joules}")

    # Make geometry
    geometry = BreakdownCircuit(
        loop_outer_diameter=0.05,
        capacitor_width=0.02,
        conductor_plate_thickness=0.005,
        dielectric_thickness=plate_separation,
        capacitance=capacitance,
        capacitor_dielectric_name=capacitor_dielectric_name,
        cable_diameter=0.0025,
        cable_insulator_thickness=0.0025,
        resistor_length=0.01,
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
    voxelizer = Build123DVoxelize()
    electric_field_update = ElectricFieldUpdate()
    magnetic_field_update = MagneticFieldUpdate()
    boundary_conditions = EMSetBoundary()
    set_capacitor_electric_field = SetCapacitorElectricFeild()
    update_position = PositionUpdate()
    update_velocity = BorisVelocityUpdate()
    deposit_charge = DepositCharge()
    charge_conservation = ChargeConservation()
    initialize_particles = InitializeParticles()

    # Make particle arrays
    particle_position = wp.zeros((3, nr_particles), wp.float32)
    particle_velocity = wp.zeros((3, nr_particles), wp.float32)
    particle_id = wp.zeros((1, nr_particles), wp.uint8)
    particle_mass_mapping = wp.from_numpy(np.array([1.0, 1000.0 * electron.mass, proton.mass], dtype=np.float32), wp.float32)
    #particle_mass_mapping = wp.from_numpy(np.array([1.0, electron.mass, electron.mass], dtype=np.float32), wp.float32)
    #particle_mass_mapping = wp.from_numpy(np.array([1.0, proton.mass / 100.0, proton.mass / 100.0], dtype=np.float32), wp.float32)
    particle_charge_mapping = wp.from_numpy(np.array([0.0, electron.charge, proton.charge], dtype=np.float32), wp.float32)

    # Make electromagnetic arrays
    material_id = wp.zeros(nr_voxels, wp.uint8)
    electric_field = wp.zeros((3, *nr_voxels), wp.float32)
    magnetic_field = wp.zeros((3, *nr_voxels), wp.float32)
    impressed_current = wp.zeros((3, *nr_voxels), wp.float32)
    eps_mapping = wp.from_numpy(np.array([m.eps for m in materials], dtype=np.float32), wp.float32)
    mu_mapping = wp.from_numpy(np.array([m.mu for m in materials], dtype=np.float32), wp.float32)
    sigma_e_mapping = wp.from_numpy(np.array([m.sigma_e for m in materials], dtype=np.float32), wp.float32)
    sigma_m_mapping = wp.from_numpy(np.array([m.sigma_m for m in materials], dtype=np.float32), wp.float32)

    # Get finale conductivity of anode resistor
    final_sigma_e = [m.sigma_e for m in materials][anode_resistor_id]
    print(f"Final sigma_e: {final_sigma_e}")

    # Set conductivity to zero in beginning
    sigma_e_mapping = sigma_e_mapping.numpy() # TODO Hacky
    sigma_e_mapping[anode_resistor_id] = 0.0
    sigma_e_mapping = wp.from_numpy(sigma_e_mapping)

    # Set initial conditions
    lower_bound = np.array([-0.0325, -0.0025, 0.0075])
    upper_bound = np.array([lower_bound[0] + particle_box, lower_bound[1] + particle_box, lower_bound[2] + particle_box])
    particle_position, particle_velocity, particle_id = initialize_particles(
        particle_position,
        particle_velocity,
        particle_id,
        lower_bound,
        upper_bound,
    )

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
    plt_freq = int(0.2e-9 / dt)
    print(f"plt_freq: {plt_freq}")
    nr_steps = int(tau_end / dt)

    # Put progress bar in 1e-9 steps
    for step in tqdm(range(nr_steps)):


        # if t < tau then charge capacitor
        if t < tau:
            alpha = (4.0 / tau)**2
            e_x = electric_field_strength * np.exp(-alpha * (t - tau)**2)
            electric_field = set_capacitor_electric_field(
                electric_field,
                material_id,
                dielectric_material_id,
                field_strength=e_x,
            )

        # if tau_switch_start < t < tau_switch_end then switches sigma_e will be increased
        if (t >= tau_switch_start) and (t < tau_switch_end):
            alpha = (4.0 / (tau_switch_end - tau_switch_start))**2
            new_sigma_e = final_sigma_e * np.exp(-alpha * (t - tau_switch_end)**2)
            sigma_e_mapping = sigma_e_mapping.numpy()
            sigma_e_mapping[anode_resistor_id] = new_sigma_e
            sigma_e_mapping = wp.from_numpy(sigma_e_mapping)

        if (t >= tau_switch_start):

            # Plot
            if step % plt_freq == 0:
                save_vtk(
                    particle_position,
                    particle_velocity,
                    particle_id,
                    electric_field,
                    magnetic_field,
                    impressed_current,
                    material_id,
                    save_dir,
                    step,
                    origin,
                    spacing,
                )

            # Zero fields
            impressed_current.zero_()

            # Update position
            particle_position = update_position(
                particle_position,
                particle_velocity,
                particle_id,
                impressed_current,
                particle_mass_mapping,
                particle_charge_mapping,
                material_id,
                origin,
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


