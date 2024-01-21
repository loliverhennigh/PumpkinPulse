from __future__ import annotations

from typing import Literal, Union

from build123d import *
from build123d import tuplify

from dense_plasma_focus.material import Material, QUARTZ, COPPER

class Resistor(Compound):
    """
    Resistor for a given curve
    """

    def __init__(
        self,
        diameter: float,
        insulator_thickness: float,
        length: float,
        resistance: float,
        insulator_material: Material = QUARTZ,
    ):

        # resistor core
        resistor_core = Circle(diameter / 2)
        resistor_core = extrude(resistor_core, length)
        resistor_core.label = "resistor_core"
        resistor_core.material = COPPER
        resistor_core.color = Color(COPPER.color)

        # resistor insulator
        resistor_insulator = Circle((diameter / 2) + insulator_thickness) - Circle(diameter / 2)
        resistor_insulator = extrude(resistor_insulator, length)
        resistor_insulator.label = "resistor_insulator"
        resistor_insulator.material = insulator_material
        resistor_insulator.color = Color(insulator_material.color)

        # Call super constructor
        super().__init__(
            label="resistor",
            children=[
                resistor_core,
                resistor_insulator,
            ],
        )

        # Make front and back connectors
        RigidJoint(
            label="front_connector",
            to_part=self,
            joint_location=Location((0, 0, length), (0, 180, 0)),
        )
        RigidJoint(
            label="back_connector",
            to_part=self,
            joint_location=Location((0, 0, 0), (0, 0, 0)),
        )