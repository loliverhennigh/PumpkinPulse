# Base class for voxelizing geometries
import numpy as np
import warp as wp

from pumpkin_pulse.data.field import Fielduint8
from pumpkin_pulse.operator.voxelize.sdf import SignedDistanceFunction
from pumpkin_pulse.operator.geometry.geometry import Geometry
from pumpkin_pulse.operator.geometry.circuit.circuit import Circuit

class Capacitor(Circuit):
    """
    Base class for initializing a capacitor
    """

    def __init__(
        self,
        capacitor_width: float,
        conductor_plate_thickness: float,
        dielectric_thickness: float,
        cable_radius: float,
        insulator_thickness: float,
        conductor_id: int,
        dielectric_id: int,
        insulator_id: int,
        center: tuple,
        centeraxis: tuple = (0.0, 1.0, 0.0),
        angle: float = 0.0,
    ):

        # Make sdf functions
        dielectric_sdf = SignedDistanceFunction.box(
            center,
            wp.vec3(
                capacitor_width/2.0,
                dielectric_thickness/2.0,
                capacitor_width/2.0,
            ),
        )
        box_conductor_sdf = SignedDistanceFunction.box(
            center,
            wp.vec3(
                capacitor_width/2.0,
                dielectric_thickness/2.0 + conductor_plate_thickness,
                capacitor_width/2.0
            ),
        )
        cylinder_conductor_sdf = SignedDistanceFunction.cylinder(
            center,
            cable_radius,
            dielectric_thickness/2.0 + conductor_plate_thickness + insulator_thickness,
        )
        conductor_sdf = SignedDistanceFunction.union(
            box_conductor_sdf,
            cylinder_conductor_sdf,
        )
        insulator_sdf = SignedDistanceFunction.box(
            center,
            wp.vec3(
                capacitor_width/2.0 + insulator_thickness,
                dielectric_thickness/2.0 + conductor_plate_thickness + insulator_thickness,
                capacitor_width/2.0 + insulator_thickness,
            ),
        )

        # Transform sdf functions
        dielectric_sdf = SignedDistanceFunction.rotate(
            dielectric_sdf,
            center,
            centeraxis,
            angle,
        )
        conductor_sdf = SignedDistanceFunction.rotate(
            conductor_sdf,
            center,
            centeraxis,
            angle,
        )
        insulator_sdf = SignedDistanceFunction.rotate(
            insulator_sdf,
            center,
            centeraxis,
            angle,
        )

        # Create operators for each sdf
        self.dielectric_operator = SignedDistanceFunction(dielectric_sdf)
        self.conductor_operator = SignedDistanceFunction(conductor_sdf)
        self.insulator_operator = SignedDistanceFunction(insulator_sdf)

        # Store ids
        self.conductor_id = conductor_id
        self.dielectric_id = dielectric_id
        self.insulator_id = insulator_id

        # Get input and output points and normals
        input_point = np.array([
            center[0],
            center[1] - (dielectric_thickness/2.0 + conductor_plate_thickness + insulator_thickness) / 2.0,
            center[2],
        ])
        input_normal = np.array([0.0, -1.0, 0.0])
        output_point = np.array([
            center[0],
            center[1] + (dielectric_thickness/2.0 + conductor_plate_thickness + insulator_thickness) / 2.0,
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
        self.conductor_operator(id_field, self.conductor_id)
        self.dielectric_operator(id_field, self.dielectric_id)

        return id_field
