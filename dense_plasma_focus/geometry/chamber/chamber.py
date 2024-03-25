from __future__ import annotations

from typing import Literal, Union

from build123d import *
from build123d import tuplify

from dense_plasma_focus.material import Material, QUARTZ, COPPER

class Chamber(Compound):
    """
    Vacuum chamber for the dense plasma focus device.
    """

    def __init__(
        self,
        feedthrough_diameter: float,
        chamber_diameter: float,
        chamber_height: float,
        chamber_thickness: float,
        material: Material = COPPER,
    ):

        # Create the cylinder for the vacuum chamber
        chamber = Location((0, 0, -chamber_thickness)) * Circle(chamber_diameter/2 + chamber_thickness)
        chamber = extrude(chamber, amount=chamber_height + 2 * chamber_thickness)

        # Subtract the feedthroughs
        feedthrough = Location((0, 0, -chamber_thickness)) * Circle(feedthrough_diameter/2)
        feedthrough = extrude(feedthrough, amount=chamber_thickness)
        chamber = chamber - feedthrough

        # Remove interior of chamber
        interior = Circle(chamber_diameter/2)
        interior = extrude(interior, amount=chamber_height)
        chamber = chamber - interior

        # Add properties to chamber
        chamber.label = "chamber"
        chamber.material = material
        chamber.color = Color(material.color)

        # Call super constructor
        super().__init__(
            label="chamber",
            children=[
                chamber,
            ],
        )

        # Make electrode joint
        RigidJoint(
            label="electrode",
            to_part=self,
            joint_location=Location((0, 0, 0), (0, 0, 0)),
        )

if __name__ == "__main__":
    # Create the chamber
    from ocp_vscode import *
    chamber = Chamber(
        feedthrough_diameter=0.5,
        chamber_diameter=2.5,
        chamber_height=2.5,
        chamber_thickness=0.25,
    )
    show(chamber, render_joints=True)
