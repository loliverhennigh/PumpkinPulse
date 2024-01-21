from __future__ import annotations

from typing import Literal, Union

from build123d import *
from build123d import tuplify

from dense_plasma_focus.material import Material, QUARTZ, COPPER

class Capacitor(Compound):
    """
    Capacitor
    """

    def __init__(
        self,
        capacitor_width: float, # assume square
        conductor_plate_thickness: float,
        dielectric_thickness: float,
        cable_diameter: float = 2.5,
        cable_insulator_thickness: float = 0.5,
        capacitance: float = 1.0,
        electrode_material: Material = COPPER,
        insulator_material: Material = QUARTZ,
    ):

        # Create conductor plate 0
        conductor_plate_0 = Rectangle(capacitor_width, capacitor_width)
        conductor_plate_0 = extrude(conductor_plate_0, amount=conductor_plate_thickness)
        conductor_plate_0.label = "conductor_plate_0"
        conductor_plate_0.material = electrode_material
        conductor_plate_0.color = Color(electrode_material.color)

        # Create conductor from plate 0 to cable
        cable_conductor_0 = Location((0, 0, -conductor_plate_thickness)) * Circle(cable_diameter/2)
        cable_conductor_0 = loft([cable_conductor_0, Rectangle(capacitor_width, capacitor_width)])
        cable_conductor_0.label = "cable_conductor_0"
        cable_conductor_0.material = electrode_material
        cable_conductor_0.color = Color(electrode_material.color)

        # Create dielectric
        dielectric = Location((0, 0, conductor_plate_thickness)) * Rectangle(capacitor_width, capacitor_width)
        dielectric = extrude(dielectric, amount=dielectric_thickness)
        dielectric.label = "dielectric"
        dielectric.material = insulator_material
        dielectric.color = Color(insulator_material.color)

        # Create conductor plate 1
        conductor_plate_1 = Location((0, 0, conductor_plate_thickness + dielectric_thickness)) * Rectangle(capacitor_width, capacitor_width)
        conductor_plate_1 = extrude(conductor_plate_1, amount=conductor_plate_thickness)
        conductor_plate_1.label = "conductor_plate_1"
        conductor_plate_1.material = electrode_material
        conductor_plate_1.color = Color(electrode_material.color)

        # Create conductor from plate 1 to cable
        cable_conductor_1 = Location((0, 0, 3 * conductor_plate_thickness + dielectric_thickness)) * Circle(cable_diameter/2)
        cable_conductor_1 = loft([cable_conductor_1, Location((0, 0, 2 * conductor_plate_thickness + dielectric_thickness)) * Rectangle(capacitor_width, capacitor_width)])
        cable_conductor_1.label = "cable_conductor_1"
        cable_conductor_1.material = electrode_material
        cable_conductor_1.color = Color(electrode_material.color)

        # Create insulator
        insulator = Location((0, 0, -conductor_plate_thickness)) * Rectangle(capacitor_width + cable_insulator_thickness, capacitor_width + cable_insulator_thickness)
        insulator = extrude(insulator, amount=4 * conductor_plate_thickness + dielectric_thickness)
        insulator = insulator - (conductor_plate_0 + cable_conductor_0 + dielectric + conductor_plate_1 + cable_conductor_1)
        insulator.label = "insulator"
        insulator.material = insulator_material
        insulator.color = Color(insulator_material.color)

        # Call super constructor
        super().__init__(
            label="capacitor",
            children=[
                conductor_plate_0,
                cable_conductor_0,
                dielectric,
                conductor_plate_1,
                cable_conductor_1,
                insulator,
            ],
        )

        # Make electrode joint
        RigidJoint(
            label="front_connector",
            to_part=self,
            joint_location=Location((0, 0, 3 * conductor_plate_thickness + dielectric_thickness), (0, 180, 0)),
        )

        ## Make anode cable joint
        #RigidJoint(
        #    label="back_connector",
        #    to_part=self,
        #    joint_location=Location((cathode_outer_diameter/2 - cable_diameter/2, 0, feedthrough_length + 4.0 * cable_diameter), (0, 0, 0)),
        #)
