import os
import numpy as np
import pyvista as pv

from pumpkin_pulse.struct.material_properties import MaterialProperties
from pumpkin_pulse.operator.operator import Operator

class MaterialPropertiesSaver(Operator):

    def __call__(
        self,
        material_properties: MaterialProperties,
        filename: str,
    ):

        # Save grid
        cell_centered_origin = material_properties.origin - material_properties.spacing * (material_properties.nr_ghost_cells - 0.5)
        x_linespace = np.linspace(
            cell_centered_origin[0],
            cell_centered_origin[0] + material_properties.spacing[0] * material_properties.shape[0],
            material_properties.shape[0],
            endpoint=False,
        )
        y_linespace = np.linspace(
            cell_centered_origin[1],
            cell_centered_origin[1] + material_properties.spacing[1] * material_properties.shape[1],
            material_properties.shape[1],
            endpoint=False,
        )
        z_linespace = np.linspace(
            cell_centered_origin[2],
            cell_centered_origin[2] + material_properties.spacing[2] * material_properties.shape[2],
            material_properties.shape[2],
            endpoint=False,
        )
        grid = pv.RectilinearGrid(
            x_linespace,
            y_linespace,
            z_linespace,
        )
        grid["solid_id"] = material_properties.id.numpy().flatten(order="F")
        grid.save(filename)

        return None
