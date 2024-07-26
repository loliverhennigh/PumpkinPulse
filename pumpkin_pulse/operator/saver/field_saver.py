import os
import numpy as np
import pyvista as pv
from typing import Union, Dict

from pumpkin_pulse.data.field import Fielduint8, Fieldfloat32, Fieldint32
from pumpkin_pulse.operator.operator import Operator

class FieldSaver(Operator):

    def __call__(
        self,
        fields: Dict[str, Union[Fielduint8, Fieldfloat32, Fieldint32]],
        filename: str,
    ):

        # Check if fields is a list
        if not isinstance(fields, dict):
            fields = {"field": fields}

        # Get the first field
        field = fields[list(fields.keys())[0]]

        # Create grid
        cell_centered_origin = field.origin
        x_linespace = np.linspace(
            cell_centered_origin[0],
            cell_centered_origin[0] + field.spacing[0] * field.shape[0],
            field.shape[0] + 1,
        )
        y_linespace = np.linspace(
            cell_centered_origin[1],
            cell_centered_origin[1] + field.spacing[1] * field.shape[1],
            field.shape[1] + 1,
        )
        z_linespace = np.linspace(
            cell_centered_origin[2],
            cell_centered_origin[2] + field.spacing[2] * field.shape[2],
            field.shape[2] + 1,
        )
        grid = pv.RectilinearGrid(
            x_linespace,
            y_linespace,
            z_linespace,
        )

        # Add fields to grid
        for name, field in fields.items():
            if field.data.shape[0] == 1:
                grid.cell_data[name] = field.data.numpy().flatten(order="F")
            elif field.data.shape[0] == 3:
                np_field = field.data.numpy()
                grid.cell_data[name + "_x"] = np_field[0].flatten(order="F")
                grid.cell_data[name + "_y"] = np_field[1].flatten(order="F")
                grid.cell_data[name + "_z"] = np_field[2].flatten(order="F")
            else:
                np_field = field.data.numpy()
                for i in range(field.data.shape[0]):
                    grid.cell_data[name + f"_{i}"] = np_field[i].flatten(order="F")

        # Save grid
        grid.save(filename)

        return None
