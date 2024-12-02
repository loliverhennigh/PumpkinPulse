# This file contains the base class for EM solver

import os
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from build123d import Compound
import numpy as np
import warp as wp
from tqdm import tqdm
from build123d import Rectangle, extrude, Location

wp.init()

from pumpkin_pulse.material import VACUUM
from pumpkin_pulse.solid import Solid
from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.operator.electromagnetism.yee_cell import (
    YeeElectricFieldUpdate,
    YeeMagneticFieldUpdate,
)
from pumpkin_pulse.operator.mesh.build123d_to_mesh import Build123DToMesh
from pumpkin_pulse.operator.mesh.mesh_to_id_field import MeshToIdField

class EMSolver:
    """
    Base Solver class for electromagnetic simulations.
    """

    def __init__(
        self,
        solids: List[Solid] = [],
        constructor: Constructor = Constructor(),
        checkpoint_dir: str = "checkpoint",
        spacing: Union[float, Tuple[float, float, float]] = 1.0,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        shape: Tuple[int, int, int] = (128, 128, 128),
        simulation_time: float = 1.0,
        compute_statistics_frequency: int = 16,
        save_fields_frequency: int = 32,
    ):
        super().__init__(
            solids=solids,
            constructor=constructor,
            checkpoint_dir=checkpoint_dir,
            spacing=spacing,
            origin=origin,
            shape=shape,
            simulation_time=simulation_time,
            compute_statistics_frequency=compute_statistics_frequency,
            save_fields_frequency=save_fields_frequency,
        )

        # Create checkpoint directory
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Temporal properties
        self.c = 299792458.0  # Speed of light in m/s
        self.dt = np.min(self.spacing) / (self.c * np.sqrt(3))  # Time step in seconds
        self.nr_time_steps = int(simulation_time / self.dt)

        # Make electromagnetic arrays
        self.electric_field = constructor.create_field(
            dtype=wp.float32,
            cardinality=3,
            shape=shape,
            origin=origin,
            spacing=spacing,
        )
        self.fields["electric_field"] = self.electric_field
        self.magnetic_field = constructor.create_field(
            dtype=wp.float32,
            cardinality=3,
            shape=shape,
            origin=origin,
            spacing=spacing,
        )
        self.fields["magnetic_field"] = self.magnetic_field
        self.charge_density = constructor.create_field(
            dtype=wp.float32,
            cardinality=1,
            shape=shape,
            origin=origin,
            spacing=spacing,
        )
        self.fields["charge_density"] = self.charge_density

        # Make solid property mappings
        self.eps_mapping = wp.zeros(self.nr_solids, wp.float32)
        self.mu_mapping = wp.zeros(self.nr_solids, wp.float32)
        self.sigma_e_mapping = wp.zeros(self.nr_solids, wp.float32)
        self.sigma_m_mapping = wp.zeros(self.nr_solids, wp.float32)

        # Make arrays for storing surface and volume charge densities
        self.total_surface_charge = wp.zeros(self.nr_solids, wp.float32)
        self.total_volume_charge = wp.zeros(self.nr_solids, wp.float32)

        # Make operators
        self.electric_field_update = YeeElectricFieldUpdate()
        self.operators["electric_field_update"] = self.electric_field_update
        self.magnetic_field_update = YeeMagneticFieldUpdate()
        self.operators["magnetic_field_update"] = self.magnetic_field_update

        # Initialize lists for energy computations
        self.statistics_time = []
        self.electric_energy = []
        self.magnetic_energy = []
        self.particle_energy = []
        self.total_energy = []
        self.total_surface_charge_store = []
        self.total_volume_charge_store = []

    def update_material_properties(self):
        """
        Update the material properties in the solver.
        """

        # Check if we need to update properties
        if self.time == 0.0 or any(
            [not solid.material.constant_conductivity for solid in self.solids]
        ):
            # Create np mappings
            eps_mapping = np.zeros(self.nr_solids, np.float32)
            mu_mapping = np.zeros(self.nr_solids, np.float32)
            sigma_e_mapping = np.zeros(self.nr_solids, np.float32)
            sigma_m_mapping = np.zeros(self.nr_solids, np.float32)

            for i, solid in enumerate(self.solids):
                material = solid.material
                eps_mapping[i] = material.eps
                mu_mapping[i] = material.mu
                if material.constant_conductivity:
                    sigma_e_mapping[i] = material.sigma_e
                else:
                    sigma_e_mapping[i] = material.sigma_e(self.time)
                sigma_m_mapping[i] = material.sigma_m

            # Update properties if necessary
            self.eps_mapping = wp.from_numpy(eps_mapping, wp.float32)
            self.mu_mapping = wp.from_numpy(mu_mapping, wp.float32)
            self.sigma_e_mapping = wp.from_numpy(sigma_e_mapping, wp.float32)
            self.sigma_m_mapping = wp.from_numpy(sigma_m_mapping, wp.float32)

    def initialize_electromagnetic_fields(self):
        """
        Initialize the electromagnetic fields.
        """

        print("Initializing electromagnetic fields...")

        # Check if any solid requires initialization
        if not any([solid.has_initial_electric_field for solid in self.solids]):
            return

        # Get number of steps for relaxation initialization (10 times the longest dimension)
        initialization_steps = 100 * int(np.max(self.shape))

        # Slowly initialize fields following relaxation
        relaxation_factors = []
        initial_e = []
        max_e = []
        for i in tqdm(range(initialization_steps)):

            # Compute relaxation factor
            if i < int(initialization_steps / 2):  # Half of the steps for rampup
                alpha = (4.0 / (initialization_steps / 2.0)) ** 2
                relaxation_factor = np.exp(-alpha * (i - (initialization_steps / 2)) ** 2)
            else:  # Half of the steps for rampdown
                relaxation_factor = 1.0
            relaxation_factors.append(relaxation_factor)

            # Set electric field for needed solids
            for id, solid in enumerate(self.solids):
                if solid.has_initial_electric_field:
                    initial_electric_field = (
                        solid.initial_electric_field[0] * relaxation_factor
                    )
                    self.electric_field = self.set_electric_field(
                        self.electric_field,
                        self.solid_id,
                        id_number=id + 1,
                        e=initial_electric_field,
                        dim=solid.initial_electric_field[1],
                        nr_ghost_cells=self.nr_ghost_cells,
                    )

            # Update electric field
            self.electric_field = self.electric_field_update(
                self.electric_field,
                self.magnetic_field,
                self.impressed_current,
                self.solid_id,
                self.eps_mapping,
                self.sigma_e_mapping,
                self.spacing,
                self.dt/2.0,
            )
            self.magnetic_field = self.magnetic_field_update(
                self.magnetic_field,
                self.electric_field,
                self.solid_id,
                self.mu_mapping,
                self.sigma_m_mapping,
                self.spacing,
                self.dt/2.0,
            )

            ## Save fields
            #if i % 1000 == 0:
            #    initial_e.append(initial_electric_field)
            #    max_e.append(np.max(self.electric_field.numpy()))

            #if i % 10000 == 0:
            #    plt.plot(initial_e, label="Initial electric field")
            #    plt.plot(max_e, label="Max electric field")
            #    plt.legend()
            #    plt.savefig(os.path.join(self.checkpoint_dir, f"initial_electric_field_{str(i).zfill(9)}.png"))
            #    plt.close()

            #    plt.plot(relaxation_factors)
            #    plt.title("Relaxation factors")
            #    plt.savefig(os.path.join(self.checkpoint_dir, f"relaxation_factors_{str(i).zfill(9)}.png"))
            #    plt.close()

            #self.save_electromagnetic_fields(i)
            #if i == 100:
            #    exit()



        print("Electromagnetic fields initialized.")

    def save_solid_id(self):
        """
        Save the solid id array.
        """

        # Create grid
        x_linespace = np.linspace(
            (self.origin[0] - self.spacing[0] / 2), # Ghost cell offset
            (self.origin[0] - self.spacing[0] / 2) + self.spacing[0] * (self.shape[0] + 1), # Ghost cell
            self.shape_with_ghost_cells[0],
        )
        y_linespace = np.linspace(
            (self.origin[1] - self.spacing[1] / 2),
            (self.origin[1] - self.spacing[1] / 2) + self.spacing[1] * (self.shape[1] + 1),
            self.shape_with_ghost_cells[1],
        )
        z_linespace = np.linspace(
            (self.origin[2] - self.spacing[2] / 2),
            (self.origin[2] - self.spacing[2] / 2) + self.spacing[2] * (self.shape[2] + 1),
            self.shape_with_ghost_cells[2],
        )
        grid = pv.RectilinearGrid(
            x_linespace,
            y_linespace,
            z_linespace,
        )
        grid["solid_id"] = self.solid_id.numpy().flatten("F")
        grid.save(os.path.join(self.checkpoint_dir, "solid_id.vtk"))

    def save_electromagnetic_fields(self, i: int = 0):
        """
        Save the electromagnetic fields.
        """

        # Create grid TODO fix this
        x_linespace = np.linspace(
            (self.origin[0] - self.spacing[0] / 2), # Ghost cell offset
            (self.origin[0] - self.spacing[0] / 2) + self.spacing[0] * (self.shape[0] + 1), # Ghost cell
            self.shape_with_ghost_cells[0],
        )
        y_linespace = np.linspace(
            (self.origin[1] - self.spacing[1] / 2),
            (self.origin[1] - self.spacing[1] / 2) + self.spacing[1] * (self.shape[1] + 1),
            self.shape_with_ghost_cells[1],
        )
        z_linespace = np.linspace(
            (self.origin[2] - self.spacing[2] / 2),
            (self.origin[2] - self.spacing[2] / 2) + self.spacing[2] * (self.shape[2] + 1),
            self.shape_with_ghost_cells[2],
        )
        grid = pv.RectilinearGrid(
            x_linespace,
            y_linespace,
            z_linespace,
        )
        np_electric_field = self.electric_field.numpy()
        np_magnetic_field = self.magnetic_field.numpy()
        np_impressed_current = self.impressed_current.numpy()
        np_charge_density = self.charge_density.numpy()
        grid["electric_field"] = np_electric_field.transpose(1, 2, 3, 0).reshape(
            -1, 3, order="F"
        )
        grid["magnetic_field"] = np_magnetic_field.transpose(1, 2, 3, 0).reshape(
            -1, 3, order="F"
        )
        grid["impressed_current"] = np_impressed_current.transpose(1, 2, 3, 0).reshape(
            -1, 3, order="F"
        )
        grid["charge_density"] = np_charge_density.flatten("F")
        grid.save(
            os.path.join(self.checkpoint_dir, f"electromagnetic_fields_{str(i).zfill(9)}.vtk")
        )

    def save_particles(self, i: int = 0):
        """
        Save the plasma particles.
        """

        # Create grid
        if self.nr_particles > 0:
            grid = pv.PolyData(self.particle_position[:, :self.nr_particles].numpy().T)
            grid["particle_id"] = self.particle_id[:, :self.nr_particles].numpy().T.astype(np.float32)
            grid["particle_velocity"] = self.particle_velocity[:, :self.nr_particles].numpy().T
            grid.save(os.path.join(self.checkpoint_dir, f"particles_{str(i).zfill(9)}.vtp"))

    def compute_energy(self):
        """
        Compute the energy in the simulation.
        """

        # Compute electric energy
        self.electric_energy.append(
            self.compute_electric_energy(
                self.electric_field,
                self.solid_id,
                self.eps_mapping,
                self.spacing,
                self.nr_ghost_cells,
            )
        )
        self.magnetic_energy.append(
            self.compute_magnetic_energy(
                self.magnetic_field,
                self.solid_id,
                self.mu_mapping,
                self.spacing,
                self.nr_ghost_cells,
            )
        )
        self.particle_energy.append(
            self.compute_particle_energy(
                self.particle_velocity,
                self.particle_id,
                self.particle_mass_mapping,
                self.nr_particles,
                self.spacing,
            )
        )
        self.total_energy.append(
            self.electric_energy[-1] + self.magnetic_energy[-1] + self.particle_energy[-1]
        )

    def plot_energy(self):
        """
        Plot the electric and magnetic energy over time.
        """

        # Create plot
        fig, ax = plt.subplots()
        ax.plot(self.statistics_time, self.electric_energy, label="Electric energy")
        ax.plot(self.statistics_time, self.magnetic_energy, label="Magnetic energy")
        ax.plot(self.statistics_time, self.particle_energy, label="Particle energy")
        ax.plot(self.statistics_time, self.total_energy, label="Total energy")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Energy (J)")
        ax.legend()
        plt.savefig(os.path.join(self.checkpoint_dir, "energy.png"))
        plt.close()

    def compute_charge_density(self):
        """
        Compute the charge density in the simulation.
        """

        # Compute charge density
        self.charge_density = self.electric_field_to_charge_density(
            self.electric_field,
            self.charge_density,
            self.solid_id,
            self.eps_mapping,
            self.spacing,
        )

        # Sum charge density
        self.total_surface_charge.zero_()
        self.total_volume_charge.zero_()
        self.total_surface_charge, self.total_volume_charge = self.sum_charge_density(
            self.charge_density,
            self.solid_id,
            self.total_surface_charge,
            self.total_volume_charge,
            self.spacing,
            self.nr_ghost_cells,
        )

        # Store charge densities
        self.total_surface_charge_store.append(self.total_surface_charge.numpy())
        self.total_volume_charge_store.append(self.total_volume_charge.numpy())

    def plot_charge_density(self):
        """
        Plot the surface and volume charge densities over time.
        """

        # Create plot
        fig, ax = plt.subplots()
        for i in range(self.nr_solids):
            ax.plot(
                self.statistics_time,
                [charge[i + 1] for charge in self.total_surface_charge_store],
                label=f"Surface charge {self.solids[i].geometry.label}",
            )
            ax.plot(
                self.statistics_time,
                [charge[i + 1] for charge in self.total_volume_charge_store],
                label=f"Volume charge {self.solids[i].geometry.label}",
            )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Charge (C)")
        ax.legend()
        plt.savefig(os.path.join(self.checkpoint_dir, "charge_density.png"))
        plt.close()

    def step(self):
        """
        Perform one time step of the simulation.
        """

        # Zero fields
        self.impressed_current.zero_()

        # Update position
        particle_position = self.position_update(
            self.particle_position,
            self.particle_velocity,
            self.particle_id,
            self.impressed_current,
            self.particle_mass_mapping,
            self.particle_charge_mapping,
            self.solid_id,
            self.permiable_mapping,
            self.origin,
            self.spacing,
            self.dt,
        )

        # Update velocity
        self.particle_velocity = self.boris_velocity_update(
            self.particle_position,
            self.particle_velocity,
            self.particle_id,
            self.electric_field,
            self.magnetic_field,
            self.particle_mass_mapping,
            self.particle_charge_mapping,
            self.origin,
            self.spacing,
            self.dt,
        )

        # Update electric field
        self.electric_field = self.electric_field_update(
            self.electric_field,
            self.magnetic_field,
            self.impressed_current,
            self.solid_id,
            self.eps_mapping,
            self.sigma_e_mapping,
            self.spacing,
            self.dt,
        )

        # Update magnetic field
        self.magnetic_field = self.magnetic_field_update(
            self.magnetic_field,
            self.electric_field,
            self.solid_id,
            self.mu_mapping,
            self.sigma_m_mapping,
            self.spacing,
            self.dt,
        )


    def run(self):
        """
        Run the simulation.
        """

        # Update material properties
        self.update_material_properties()

        # Update plasma properties
        self.update_plasma_properties()

        # Initialize solid id
        self.initialize_solid_id()

        # Save solid id
        self.save_solid_id()

        # Initialize electromagnetic fields
        self.initialize_electromagnetic_fields()

        # Save electromagnetic fields
        self.save_electromagnetic_fields()

        # Initialize plasma
        self.initialize_plasma()

        # Save particles
        self.save_particles()

        # Run simulation
        for i in tqdm(range(self.nr_time_steps)):

            # Get time
            self.time = i * self.dt

            # Save statistics if necessary
            if i % self.compute_statistics_frequency == 0:
                self.compute_energy()
                self.compute_charge_density()
                self.statistics_time.append(self.time)

            # Save fields if necessary
            if i % self.save_fields_frequency == 0:
                self.plot_energy()
                self.plot_charge_density()

            # Save particles if necessary
            if i % self.save_fields_frequency == 0:
                self.save_particles(i)
                self.save_electromagnetic_fields(i)

            # Update material properties
            self.update_material_properties()

            # Take a step
            self.step()


