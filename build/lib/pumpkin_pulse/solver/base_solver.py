# This file contains the base class for all solvers

from typing import List, Tuple, Union
import os
import numpy as np
import warp as wp
from build123d import extrude, Rectangle, Location

from pumpkin_pulse.material import VACUUM
from pumpkin_pulse.solid import Solid
from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.operator.saver.field_saver import FieldSaver
from pumpkin_pulse.operator.mesh.build123d_to_mesh import Build123DToMesh
from pumpkin_pulse.operator.mesh.mesh_to_id_field import MeshToIdField

class BaseSolver:
    """
    Base Solver class for all solvers.
    """

    def __init__(
        self,
        constructor: Constructor = Constructor(),
        checkpoint_dir: str = "checkpoint",
        spacing: Union[float, Tuple[float, float, float]] = 1.0,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        shape: Tuple[int, int, int] = (128, 128, 128),
        simulation_time: float = 1.0,
        compute_statistics_frequency: int = 16,
        save_fields_frequency: int = 32,
    ):

        # Set parameters
        self.solids = solids
        self.constructor = constructor
        self.checkpoint_dir = checkpoint_dir
        self.spacing = spacing
        self.origin = origin
        self.shape = shape
        self.simulation_time = simulation_time
        self.compute_statistics_frequency = compute_statistics_frequency
        self.save_fields_frequency = save_fields_frequency

        # Add vacuum as default solid that fills the domain
        vacuum = Rectangle(shape[0] * spacing[0], shape[1] * spacing[1])
        vacuum = extrude(vacuum, shape[2] * spacing[2])
        vacuum = Location((origin[0] + shape[0] * spacing[0] / 2, origin[1] + shape[1] * spacing[1] / 2, origin[2])) * vacuum
        vacuum = Solid(vacuum, VACUUM, "BackgroundVacuum", permiable=True)
        self.solids = [vacuum] + self.solids

        # Get number of solids
        self.num_solids = len(self.solids)

        # Create checkpoint directory
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Make fields and operators
        self.fields = {}
        self.operators = {}

        # Set time 
        self.time = 0.0
        self.step_number = 0
        
        # Make operators
        self.field_saver = FieldSaver()
        self.operators["field_saver"] = self.field_saver
        self.build123d_to_mesh = Build123DToMesh()
        self.operators["build123d_to_mesh"] = self.build123d_to_mesh
        self.mesh_to_id_field = MeshToIdField()
        self.operators["mesh_to_id_field"] = self.mesh_to_id_field

        # Make cell centered solid ids
        self.id_field = constructor.create_field(
            dtype=wp.uint8,
            cardinality=1,
            shape=shape,
            origin=origin,
            spacing=spacing,
        )
        self.fields["id_field"] = self.id_field

    def save_fields(self, field_names):
        for field_name in field_names:
            if field_name in self.fields:
                self.field_saver(
                    self.fields[field_name],
                    os.path.join(self.checkpoint_dir, f"{field_name}_{self.step_number}.vtk"),
                )

    def set_id_field(self):
        """
        Set the id field.
        """

        # Run through solids
        for i, solid in enumerate(self.solids):

            # Make mesh
            mesh = self.build123d_to_mesh(solid.geometry)

            # Set id field
            self.id_field = self.mesh_to_id_field(
                mesh,
                self.id_field,
                i,
            )

    def step(self):
        pass

    def run(self):
        pass
