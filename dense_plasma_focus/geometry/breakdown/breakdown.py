from __future__ import annotations

from typing import Literal, Union

from build123d import *
from build123d import tuplify

from dense_plasma_focus.material import Material, QUARTZ, COPPER

class Breakdown(Compound):
    """
    Parrallel plates to test breakdown voltage
    """

    def __init__(
        self,
        breakdown_width: float, # assume square
        breakdown_plate_thickness: float,
        breakdown_thickness: float,
        cable_diameter: float = 2.5,
        electrode_material: Material = COPPER,
    ):

        # Create conductor plate 0
        conductor_plate_0 = Rectangle(breakdown_width, breakdown_width)
        conductor_plate_0 = extrude(conductor_plate_0, amount=breakdown_plate_thickness)
        conductor_plate_0.label = "conductor_plate_0"
        conductor_plate_0.material = electrode_material
        conductor_plate_0.color = Color(electrode_material.color)

        # Create conductor from plate 0 to cable
        cable_conductor_0 = Location((0, 0, -breakdown_plate_thickness)) * Circle(cable_diameter/2)
        cable_conductor_0 = loft([cable_conductor_0, Rectangle(breakdown_width, breakdown_width)])
        cable_conductor_0.label = "cable_conductor_0"
        cable_conductor_0.material = electrode_material
        cable_conductor_0.color = Color(electrode_material.color)

        # Create conductor plate 1
        conductor_plate_1 = Location((0, 0, breakdown_thickness + breakdown_plate_thickness)) * Rectangle(breakdown_width, breakdown_width)
        conductor_plate_1 = extrude(conductor_plate_1, amount=breakdown_plate_thickness)
        conductor_plate_1.label = "conductor_plate_1"
        conductor_plate_1.material = electrode_material
        conductor_plate_1.color = Color(electrode_material.color)

        # Create conductor from plate 1 to cable
        cable_conductor_1 = Location((0, 0, breakdown_thickness + 3 * breakdown_plate_thickness)) * Circle(cable_diameter/2)
        cable_conductor_1 = loft([cable_conductor_1, Location((0, 0, 2 * breakdown_plate_thickness + breakdown_thickness)) * Rectangle(breakdown_width, breakdown_width)])
        cable_conductor_1.label = "cable_conductor_1"
        cable_conductor_1.material = electrode_material
        cable_conductor_1.color = Color(electrode_material.color)

        # Call super constructor
        super().__init__(
            label="breakdown",
            children=[
                conductor_plate_0,
                cable_conductor_0,
                conductor_plate_1,
                cable_conductor_1,
            ],
        )

        # Make electrode joint
        RigidJoint(
            label="front_connector",
            to_part=self,
            joint_location=Location((0, 0, 3 * breakdown_plate_thickness + breakdown_thickness), (0, 180, 0)),
        )

        # Make anode cable joint
        RigidJoint(
            label="back_connector",
            to_part=self,
            joint_location=Location((0, 0, -1*breakdown_plate_thickness), (0, 0, 180)),
        )
