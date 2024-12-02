# Chamber

import os
import numpy as np
import warp as wp
import matplotlib.pyplot as plt
import dataclasses
import itertools
from tqdm import tqdm

wp.init()

from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.data import Fielduint8, Fieldfloat32
from pumpkin_pulse.operator import Operator
from pumpkin_pulse.operator.voxelize import (
    Tube,
)
from pumpkin_pulse.operator.saver import FieldSaver


class Chamber(Operator):
    def __init__(
        self,
        chamber_wall_thickness: float,
        interaction_radius: float,
        interaction_bounds: float,
        acceleration_bounds: float,
        formation_radius: float,
        formation_bounds: float,
        diverter_inlet_radius: float,
        diverter_inlet_bounds: float,
        diverter_radius: float,
        diverter_bounds: float,
        vacuum_id: int,
        chamber_id: int,
    ):

        # Chamber parameters
        self.chamber_wall_thickness = chamber_wall_thickness
        self.interaction_radius = interaction_radius
        self.interaction_bounds = interaction_bounds
        self.acceleration_bounds = acceleration_bounds
        self.formation_radius = formation_radius
        self.formation_bounds = formation_bounds
        self.diverter_inlet_radius = diverter_inlet_radius
        self.diverter_inlet_bounds = diverter_inlet_bounds
        self.diverter_radius = diverter_radius
        self.diverter_bounds = diverter_bounds
        self.vacuum_id = vacuum_id
        self.chamber_id = chamber_id

        # Interaction section
        interaction_position_array = np.linspace(0.0, interaction_bounds, 100)
        interaction_radius_array = np.full_like(interaction_position_array, interaction_radius)
        interaction_vacuum_position_array = interaction_position_array
        interaction_vacuum_radius_array = interaction_radius_array
        interaction_chamber_position_array = interaction_position_array
        interaction_chamber_radius_array = interaction_radius_array + chamber_wall_thickness
        
        # Acceleration section
        acceleration_position_array = np.linspace(interaction_bounds, acceleration_bounds, 100)
        acceleration_radius_slope_array = (formation_radius - interaction_radius) / (acceleration_bounds - interaction_bounds)
        acceleration_radius_array = (
            acceleration_radius_slope_array * (acceleration_position_array - interaction_bounds) + interaction_radius
        )
        acceleration_vacuum_position_array = acceleration_position_array
        acceleration_vacuum_radius_array = acceleration_radius_array
        acceleration_chamber_position_array = acceleration_position_array
        acceleration_chamber_radius_array = acceleration_radius_array + chamber_wall_thickness

        # Formation section
        formation_position_array = np.linspace(acceleration_bounds, formation_bounds, 100)
        formation_radius_array = np.full_like(formation_position_array, formation_radius)
        formation_vacuum_position_array = formation_position_array
        formation_vacuum_radius_array = formation_radius_array
        formation_chamber_position_array = formation_position_array
        formation_chamber_radius_array = formation_radius_array + chamber_wall_thickness

        # Diverter inlet section
        diverter_inlet_position_array = np.linspace(formation_bounds, diverter_inlet_bounds, 100)
        diverter_inlet_radius_array = np.full_like(diverter_inlet_position_array, diverter_inlet_radius)
        diverter_inlet_vacuum_position_array = diverter_inlet_position_array
        diverter_inlet_vacuum_radius_array = diverter_inlet_radius_array
        diverter_inlet_chamber_position_array = diverter_inlet_position_array
        diverter_inlet_chamber_radius_array = diverter_inlet_radius_array + chamber_wall_thickness
        diverter_inlet_chamber_radius_array[diverter_inlet_position_array < formation_bounds + chamber_wall_thickness] = diverter_radius + chamber_wall_thickness
        diverter_inlet_chamber_radius_array[diverter_inlet_position_array > diverter_inlet_bounds - chamber_wall_thickness] = diverter_radius + chamber_wall_thickness

        # Diverter section
        diverter_vacuum_position_array = np.linspace(diverter_inlet_bounds, diverter_bounds, 100)
        diverter_vacuum_radius_array = np.full_like(diverter_vacuum_position_array, formation_radius)
        diverter_chamber_position_array = np.linspace(diverter_inlet_bounds, diverter_bounds + chamber_wall_thickness, 100)
        diverter_chamber_radius_array = np.full_like(diverter_chamber_position_array, formation_radius + chamber_wall_thickness)

        # Stack sections
        vacuum_radius = np.concatenate(
            (
                interaction_vacuum_radius_array,
                acceleration_vacuum_radius_array,
                formation_vacuum_radius_array,
                diverter_inlet_vacuum_radius_array,
                diverter_vacuum_radius_array,
            )
        )
        chamber_radius = np.concatenate(
            (
                interaction_chamber_radius_array,
                acceleration_chamber_radius_array,
                formation_chamber_radius_array,
                diverter_inlet_chamber_radius_array,
                diverter_chamber_radius_array,
            )
        )
        vacuum_position = np.concatenate(
            (
                interaction_vacuum_position_array,
                acceleration_vacuum_position_array,
                formation_vacuum_position_array,
                diverter_inlet_vacuum_position_array,
                diverter_vacuum_position_array,
            )
        )
        chamber_position = np.concatenate(
            (
                interaction_chamber_position_array,
                acceleration_chamber_position_array,
                formation_chamber_position_array,
                diverter_inlet_chamber_position_array,
                diverter_chamber_position_array,
            )
        )
        vacuum_path = np.stack(
            [
                vacuum_position,
                np.zeros_like(vacuum_position),
                np.zeros_like(vacuum_position),
            ],
            axis=1,
        )
        chamber_path = np.stack(
            [
                chamber_position,
                np.zeros_like(chamber_position),
                np.zeros_like(chamber_position),
            ],
            axis=1,
        )
    
        # Flip radius and thickness to make symmetric
        reflected_vacuum_radius = np.flip(vacuum_radius)
        reflected_chamber_radius = np.flip(chamber_radius)
        reflected_vacuum_path = -np.flip(vacuum_path, axis=0)
        reflected_chamber_path = -np.flip(chamber_path, axis=0)
        self.vacuum_radius = np.concatenate((reflected_vacuum_radius, vacuum_radius))
        self.vacuum_path = np.concatenate((reflected_vacuum_path, vacuum_path))
        self.chamber_radius = np.concatenate((reflected_chamber_radius, chamber_radius))
        self.chamber_path = np.concatenate((reflected_chamber_path, chamber_path))
    
        # Make the Tube operator
        self.tube_operator = Tube()
    
    def __call__(
        self,
        id_field: Fielduint8,
    ) -> Fielduint8:

        # Initialize insulator and then vacuum
        id_field = self.tube_operator(
            id_field,
            wp.from_numpy(self.chamber_path, dtype=wp.vec3),
            wp.from_numpy(self.chamber_radius, dtype=wp.float32),
            self.chamber_id,
        )
        id_field = self.tube_operator(
            id_field,
            wp.from_numpy(self.vacuum_path, dtype=wp.vec3),
            wp.from_numpy(self.vacuum_radius, dtype=wp.float32),
            self.vacuum_id, 
        )

        return id_field


if __name__ == "__main__":

    # Define simulation parameters
    dx = 0.00025 # 1 mm
    origin = (-0.55, -0.085, -0.085) # meters
    spacing = (dx, dx, dx)
    shape = (int(1.1 / dx), int(0.17 / dx), int(0.17 / dx))
    nr_cells = shape[0] * shape[1] * shape[2]
    print(f"Number of million cells: {nr_cells / 1e6}")

    # Chamber parameters
    chamber_wall_thickness = 0.005
    interaction_radius = 0.015
    interaction_bounds = 0.1
    acceleration_bounds = 0.3
    formation_radius = 0.025
    formation_bounds = 0.4
    diverter_inlet_radius = 0.01
    diverter_inlet_bounds = 0.45
    diverter_radius = 0.025
    diverter_bounds = 0.5
    vacuum_id = 0
    chamber_id = 1

    # Make the field saver
    field_saver = FieldSaver()

    # Make the constructor
    constructor = Constructor(
        shape=shape,
        origin=origin,
        spacing=spacing,
    )

    # Make the fields
    id_field = constructor.create_field(
        dtype=wp.uint8,
        cardinality=1,
    )

    # Make the Helion chamber
    helion_chamber_operator = HelionChamber(
        chamber_wall_thickness=chamber_wall_thickness,
        interaction_radius=interaction_radius,
        interaction_bounds=interaction_bounds,
        acceleration_bounds=acceleration_bounds,
        formation_radius=formation_radius,
        formation_bounds=formation_bounds,
        diverter_inlet_radius=diverter_inlet_radius,
        diverter_inlet_bounds=diverter_inlet_bounds,
        diverter_radius=diverter_radius,
        diverter_bounds=diverter_bounds,
        vacuum_id=vacuum_id,
        chamber_id=chamber_id,
    )

    # Run the Helion chamber
    id_field = helion_chamber_operator(id_field)

    # Save the fields
    field_saver(
        {"id_field": id_field},
        "id_field.vtk",
    )
