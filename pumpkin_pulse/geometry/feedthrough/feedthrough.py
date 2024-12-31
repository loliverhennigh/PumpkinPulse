from __future__ import annotations

from typing import Literal, Union

from build123d import *
from build123d import tuplify

from pumpkin_pulse.material import Material, QUARTZ, COPPER


class Feedthrough(Compound):
    """
    Feed through for the vacuum chamber
    """

    def __init__(
        self,
        anode_outer_diameter: float,
        cathode_inner_diameter: float,
        cathode_outer_diameter: float,
        insulator_thickness: float,
        feedthrough_length: float,
        cable_diameter: float = 2.5,
        cable_insulator_thickness: float = 0.5,
        electrode_material: Material = COPPER,
        insulator_material: Material = QUARTZ,
    ):
        # Create the anode feedthrough
        anode_feedthrough = Circle(anode_outer_diameter / 2)
        anode_feedthrough = extrude(anode_feedthrough, amount=feedthrough_length)
        anode_feedthrough.label = "anode_feedthrough"
        anode_feedthrough.material = electrode_material
        anode_feedthrough.color = Color(electrode_material.color)

        # Create loft between anode feedthrough and cable
        anode_cable = [
            Location((0, 0, feedthrough_length)) * Circle(anode_outer_diameter / 2),
            Location(
                (
                    cathode_outer_diameter / 2 - cable_diameter / 2,
                    0,
                    feedthrough_length + 4.0 * cable_diameter,
                ),
                (0, 0, 0),
            )
            * Circle(cable_diameter / 2),
        ]
        anode_cable = loft(anode_cable)
        anode_cable.label = "anode_cable"
        anode_cable.material = electrode_material
        anode_cable.color = Color(electrode_material.color)

        # Create insulator between anode and cathode
        anode_cathod_insulator = Circle(cathode_inner_diameter / 2)
        anode_cathod_insulator = anode_cathod_insulator - Circle(
            anode_outer_diameter / 2
        )
        anode_cathod_insulator = extrude(
            anode_cathod_insulator, amount=feedthrough_length
        )
        anode_cathod_insulator.label = "anode_cathode_insulator"
        anode_cathod_insulator.material = insulator_material
        anode_cathod_insulator.color = Color(insulator_material.color)

        # Create anode cable insulator
        anode_cable_insulator = [
            Location((0, 0, feedthrough_length)) * Circle(cathode_inner_diameter / 2),
            Location(
                (
                    cathode_outer_diameter / 2 - cable_diameter / 2,
                    0,
                    feedthrough_length + 4.0 * cable_diameter,
                ),
                (0, 0, 0),
            )
            * Circle(cable_diameter / 2 + cable_insulator_thickness),
        ]
        anode_cable_insulator = loft(anode_cable_insulator)
        anode_cable_insulator = anode_cable_insulator - anode_cable
        anode_cable_insulator.label = "anode_cable_insulator"
        anode_cable_insulator.material = insulator_material
        anode_cable_insulator.color = Color(insulator_material.color)

        # Create the cathode feedthrough
        cathode_feedthrough = Circle(cathode_outer_diameter / 2)
        cathode_feedthrough = cathode_feedthrough - Circle(cathode_inner_diameter / 2)
        cathode_feedthrough = extrude(cathode_feedthrough, amount=feedthrough_length)
        cathode_feedthrough.label = "cathode_feedthrough"
        cathode_feedthrough.material = electrode_material
        cathode_feedthrough.color = Color(electrode_material.color)

        # Create the loft between cathode feedthrough and cable
        cathode_cable = [
            Location((0, 0, feedthrough_length)) * Circle(cathode_outer_diameter / 2),
            Location(
                (
                    -cathode_outer_diameter / 2 + cable_diameter / 2,
                    0,
                    feedthrough_length + 4.0 * cable_diameter,
                ),
                (0, 0, 0),
            )
            * Circle(cable_diameter / 2),
        ]
        cathode_cable = loft(cathode_cable)
        cathode_cable = cathode_cable - (anode_cable + anode_cable_insulator)
        cathode_cable.label = "cathode_cable"
        cathode_cable.material = electrode_material
        cathode_cable.color = Color(electrode_material.color)

        # Create insulator between cathode and vacuum chamber
        cathode_vacuum_insulator = Circle(
            cathode_outer_diameter / 2 + insulator_thickness
        )
        cathode_vacuum_insulator = cathode_vacuum_insulator - Circle(
            cathode_outer_diameter / 2
        )
        cathode_vacuum_insulator = extrude(
            cathode_vacuum_insulator, amount=feedthrough_length
        )
        cathode_vacuum_insulator.label = "cathode_vacuum_insulator"
        cathode_vacuum_insulator.material = insulator_material
        cathode_vacuum_insulator.color = Color(insulator_material.color)

        # Create cathode cable insulator
        cathode_cable_insulator = [
            Location((0, 0, feedthrough_length))
            * Circle(cathode_outer_diameter / 2 + insulator_thickness),
            Location(
                (
                    -cathode_outer_diameter / 2 + cable_diameter / 2,
                    0,
                    feedthrough_length + 4.0 * cable_diameter,
                ),
                (0, 0, 0),
            )
            * Circle(cable_diameter / 2 + cable_insulator_thickness),
        ]
        cathode_cable_insulator = loft(cathode_cable_insulator)
        cathode_cable_insulator = cathode_cable_insulator - (
            anode_cable + anode_cable_insulator + cathode_cable
        )
        cathode_cable_insulator.label = "cathode_cable_insulator"
        cathode_cable_insulator.material = insulator_material
        cathode_cable_insulator.color = Color(insulator_material.color)

        # Call super constructor
        super().__init__(
            label="feedthrough",
            children=[
                anode_feedthrough,
                anode_cable,
                anode_cable_insulator,
                anode_cathod_insulator,
                cathode_feedthrough,
                cathode_cable,
                cathode_vacuum_insulator,
                cathode_cable_insulator,
            ],
        )

        # Make electrode joint
        RigidJoint(
            label="electrode",
            to_part=self,
            joint_location=Location((0, 0, 0), (0, 180, 0)),
        )

        # Make anode cable joint
        RigidJoint(
            label="anode_cable",
            to_part=self,
            joint_location=Location(
                (
                    cathode_outer_diameter / 2 - cable_diameter / 2,
                    0,
                    feedthrough_length + 4.0 * cable_diameter,
                ),
                (0, 0, 0),
            ),
        )

        # Make cathode cable joint
        RigidJoint(
            label="cathode_cable",
            to_part=self,
            joint_location=Location(
                (
                    -cathode_outer_diameter / 2 + cable_diameter / 2,
                    0,
                    feedthrough_length + 4.0 * cable_diameter,
                ),
                (0, 0, 180),
            ),
        )
