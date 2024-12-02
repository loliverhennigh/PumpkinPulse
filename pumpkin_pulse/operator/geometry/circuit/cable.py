# Base class for voxelizing geometries
import numpy as np
import warp as wp

from pumpkin_pulse.data.field import Fielduint8
from pumpkin_pulse.operator.voxelize.tube import BezierTube
from pumpkin_pulse.operator.geometry.circuit.circuit import Circuit

class Cable(Circuit):
    """
    Base class for initializing a cable between two points and normals
    """

    def __init__(
        self,
        input_point: np.ndarray,
        input_normal: np.ndarray,
        output_point: np.ndarray,
        output_normal: np.ndarray,
        cable_radius: float,
        insulator_thickness: float,
        conductor_id: int,
        insulator_id: int,
        nr_points: int=100,
        scale: float=1.0,
    ):

        # Create operator for Bezier Tube
        self.bezier_tube_operator = BezierTube()

        # Store cable parameters
        self.cable_radius = cable_radius
        self.scale = scale
        self.nr_points = nr_points
        self.insulator_thickness = insulator_thickness
        self.conductor_id = conductor_id
        self.insulator_id = insulator_id

        # Initialize parent class
        super().__init__(
            input_point,
            input_normal,
            output_point,
            output_normal,
        )


    def __call__(
        self,
        id_field: Fielduint8,
    ):

        # Create path for cable
        self.bezier_tube_operator(
            id_field,
            wp.vec3(self.input_point),
            wp.vec3(self.input_normal),
            wp.vec3(self.output_point),
            wp.vec3(self.output_normal),
            num_points=self.nr_points,
            scale=self.scale,
            radius=self.cable_radius + self.insulator_thickness,
            id_number=self.insulator_id,
        )
        self.bezier_tube_operator(
            id_field,
            wp.vec3(self.input_point),
            wp.vec3(self.input_normal),
            wp.vec3(self.output_point),
            wp.vec3(self.output_normal),
            num_points=self.nr_points,
            scale=self.scale,
            radius=self.cable_radius,
            id_number=self.conductor_id,
        )

        return id_field
