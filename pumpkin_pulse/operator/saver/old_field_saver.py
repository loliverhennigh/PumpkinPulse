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
        cell_centered_origin = np.array(field.origin)
        cell_centered_origin[0] += field.spacing[0] * field.offset[0]
        cell_centered_origin[1] += field.spacing[1] * field.offset[1]
        cell_centered_origin[2] += field.spacing[2] * field.offset[2]
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
            
            # Get numpy field
            np_field = field.data.numpy()

            # Reshape field
            if field.cardinality == 1:
                np_field = np_field.flatten(order="F")
            elif int(field.ordering) == 0: # SoA
                np_field = np.stack(
                    [np_field[i].flatten(order="F") for i in range(field.cardinality)],
                    axis=1,
                ).reshape(-1, field.cardinality, order="F")
            elif int(field.ordering) == 1: # AoS
                np_field = np_field.reshape(-1, field.cardinality, order="F")
            else:
                raise ValueError(f"Unknown ordering {field.ordering}")

            # Add field to grid
            grid.cell_data[name] = np_field

        # Save grid
        grid.save(filename)

        return None
