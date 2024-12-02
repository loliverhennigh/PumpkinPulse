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

        # Check if fields is a dictionary
        if not isinstance(fields, dict):
            fields = {"field": fields}

        # Get the first field to extract common properties
        field = fields[list(fields.keys())[0]]

        # Compute the origin adjusted for cell-centered data
        cell_centered_origin = np.array(field.origin)
        cell_centered_origin[0] += field.spacing[0] * field.offset[0]
        cell_centered_origin[1] += field.spacing[1] * field.offset[1]
        cell_centered_origin[2] += field.spacing[2] * field.offset[2]

        # Create the ImageData grid
        grid = pv.ImageData(
            dimensions=tuple(np.array(field.shape) + 1),
            spacing=tuple(field.spacing),
            origin=tuple(cell_centered_origin)
        )

        # Add fields to the ImageData
        for name, field in fields.items():
            # Get numpy field data
            np_field = field.data.numpy()

            # Reshape the field based on its cardinality and ordering
            if field.cardinality == 1:
                np_field = np_field.flatten(order="F")
            elif int(field.ordering) == 0:  # Structure of Arrays (SoA)
                np_field = np.stack(
                    [np_field[i].flatten(order="F") for i in range(field.cardinality)],
                    axis=1,
                ).reshape(-1, field.cardinality, order="F")
            elif int(field.ordering) == 1:  # Array of Structures (AoS)
                np_field = np_field.reshape(-1, field.cardinality, order="F")
            else:
                raise ValueError(f"Unknown ordering {field.ordering}")

            # Add the field to the grid as cell data
            grid.cell_data[name] = np_field

        # Save the ImageData as an image VTK file
        grid.save(filename)

        return None
