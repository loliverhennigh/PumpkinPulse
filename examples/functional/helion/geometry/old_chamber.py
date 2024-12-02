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


class HelionChamber(Operator):
    def __init__(
        self,
        chamber_wall_thickness: float,
        compressor_radius: float,
        compressor_bounds: float,
        initializer_radius: float,
        initializer_bounds: float,
        diverter_inlet_radius: float,
        diverter_inlet_bounds: float,
        diverter_radius: float,
        diverter_bounds: float,
        vacuum_id: int,
        chamber_id: int,
    ):

        # Chamber parameters
        self.chamber_wall_thickness = chamber_wall_thickness
        self.compressor_radius = compressor_radius
        self.compressor_bounds = compressor_bounds
        self.initializer_radius = initializer_radius
        self.initializer_bounds = initializer_bounds
        self.diverter_inlet_radius = diverter_inlet_radius
        self.diverter_inlet_bounds = diverter_inlet_bounds
        self.diverter_radius = diverter_radius
        self.diverter_bounds = diverter_bounds
        self.vacuum_id = vacuum_id
        self.chamber_id = chamber_id

        # Compressor section
        compressor_vacuum_position_array = np.linspace(0.0, compressor_bounds, 100)
        compressor_vacuum_radius_array = np.full_like(compressor_vacuum_position_array, compressor_radius)
        compressor_chamber_position_array = np.linspace(0.0, compressor_bounds, 100)
        compressor_chamber_radius_array = np.full_like(compressor_chamber_position_array , compressor_radius + chamber_wall_thickness)
    
        # Initializer section
        initializer_position_array = np.linspace(compressor_bounds, initializer_bounds, 100)
        initializer_radius_slope_array = (initializer_radius - compressor_radius) / (initializer_bounds - compressor_bounds)
        initializer_radius_array = (
            initializer_radius_slope_array * (initializer_position_array - compressor_bounds) + compressor_radius
        )
        initializer_vacuum_position_array = initializer_position_array
        initializer_vacuum_radius_array = initializer_radius_array
        initializer_chamber_position_array = initializer_position_array
        initializer_chamber_radius_array = initializer_radius_array + chamber_wall_thickness
    
        # Diverter inlet section
        diverter_inlet_position_array = np.linspace(initializer_bounds, diverter_inlet_bounds, 100)
        diverter_inlet_radius_array = np.full_like(diverter_inlet_position_array, diverter_inlet_radius)
        diverter_inlet_vacuum_position_array = diverter_inlet_position_array
        diverter_inlet_vacuum_radius_array = diverter_inlet_radius_array
        diverter_inlet_chamber_position_array = diverter_inlet_position_array
        diverter_inlet_chamber_radius_array = diverter_inlet_radius_array + chamber_wall_thickness
        diverter_inlet_chamber_radius_array[diverter_inlet_position_array < initializer_bounds + chamber_wall_thickness] = diverter_radius + chamber_wall_thickness
        diverter_inlet_chamber_radius_array[diverter_inlet_position_array > diverter_inlet_bounds - chamber_wall_thickness] = diverter_radius + chamber_wall_thickness
    
        # Diverter section
        diverter_vacuum_position_array = np.linspace(diverter_inlet_bounds, diverter_bounds, 100)
        diverter_vacuum_radius_array = np.full_like(diverter_vacuum_position_array, initializer_radius)
        diverter_chamber_position_array = np.linspace(diverter_inlet_bounds, diverter_bounds + chamber_wall_thickness, 100)
        diverter_chamber_radius_array = np.full_like(diverter_chamber_position_array, initializer_radius + chamber_wall_thickness)
    
        # Stack sections
        vacuum_radius = np.concatenate(
            (
                compressor_vacuum_radius_array,
                initializer_vacuum_radius_array,
                diverter_inlet_vacuum_radius_array,
                diverter_vacuum_radius_array,
            )
        )
        chamber_radius = np.concatenate(
            (
                compressor_chamber_radius_array,
                initializer_chamber_radius_array,
                diverter_inlet_chamber_radius_array,
                diverter_chamber_radius_array,
            )
        )
        vacuum_position = np.concatenate(
            (
                compressor_vacuum_position_array,
                initializer_vacuum_position_array,
                diverter_inlet_vacuum_position_array,
                diverter_vacuum_position_array,
            )
        )
        chamber_position = np.concatenate(
            (
                compressor_chamber_position_array,
                initializer_chamber_position_array,
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
    dx = 0.0005 # 1 mm
    origin = (-0.55, -0.085, -0.085) # meters
    spacing = (dx, dx, dx)
    shape = (int(1.1 / dx), int(0.17 / dx), int(0.17 / dx))
    nr_cells = shape[0] * shape[1] * shape[2]
    print(f"Number of million cells: {nr_cells / 1e6}")

    # Chamber parameters
    chamber_wall_thickness = 0.005
    compressor_radius = 0.020
    compressor_bounds = 2.0 * compressor_radius
    initializer_radius = 0.05
    initializer_bounds = 0.35
    diverter_inlet_radius = 0.015
    diverter_inlet_bounds = 0.4
    diverter_radius = initializer_radius
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
        compressor_radius=compressor_radius,
        compressor_bounds=compressor_bounds,
        initializer_radius=initializer_radius,
        initializer_bounds=initializer_bounds,
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
