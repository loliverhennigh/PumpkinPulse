# Base class for voxelizing geometries
import numpy as np
import warp as wp

from pumpkin_pulse.data.field import Fielduint8
from pumpkin_pulse.operator.voxelize.sdf import SignedDistanceFunction
from pumpkin_pulse.operator.geometry.geometry import Geometry
from pumpkin_pulse.operator.geometry.circuit.circuit import Circuit

class Element(Circuit):
    """
    Base class for initializing a capacitor
    """

    
    def __init__(
        self,
        cable_radius: float,
        insulator_thickness: float,
        element_length: float,
        element_id: int,
        insulator_id: int,
        center: tuple,
        centeraxis: tuple = (0.0, 1.0, 0.0),
        angle: float = 0.0,
    ):

        # Make sdf functions
        insulator_sdf = SignedDistanceFunction.cylinder(
            center,
            cable_radius + insulator_thickness,
            element_length/2.0,
        )
        element_sdf = SignedDistanceFunction.cylinder(
            center,
            cable_radius,
            element_length/2.0,
        )

        # Rotate the sdf functions
        insulator_sdf = SignedDistanceFunction.rotate(
            insulator_sdf,
            center,
            centeraxis,
            angle,
        )
        element_sdf = SignedDistanceFunction.rotate(
            element_sdf,
            center,
            centeraxis,
            angle,
        )

        # Create operators for each sdf
        self.element_operator = SignedDistanceFunction(element_sdf)
        self.insulator_operator = SignedDistanceFunction(insulator_sdf)

        # Store ids
        self.element_id = element_id
        self.insulator_id = insulator_id

        # Store input and output points and normals
        input_point = np.array([
            center[0],
            center[1] - element_length/2.0,
            center[2],
        ])
        input_normal = np.array([0.0, -1.0, 0.0])
        output_point = np.array([
            center[0],
            center[1] + element_length/2.0,
            center[2],
        ])
        output_normal = np.array([0.0, 1.0, 0.0])

        # Rotate input and output points and normals
        input_point = Geometry._rotate_point(
            input_point,
            center,
            centeraxis,
            angle,
        )
        input_normal = Geometry._rotate_point(
            input_normal,
            center,
            centeraxis,
            angle,
        )
        output_point = Geometry._rotate_point(
            output_point,
            center,
            centeraxis,
            angle,
        )
        output_normal = Geometry._rotate_point(
            output_normal,
            center,
            centeraxis,
            angle,
        )

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

        # Apply operators
        self.insulator_operator(id_field, self.insulator_id)
        self.element_operator(id_field, self.element_id)

        return id_field
