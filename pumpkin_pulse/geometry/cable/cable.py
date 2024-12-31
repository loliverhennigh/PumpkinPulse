from __future__ import annotations

from typing import Literal, Union

from build123d import *
from build123d import tuplify

from pumpkin_pulse.material import Material, QUARTZ, COPPER


class Cable(Compound):
    """
    Cable for a given curve
    """

    def __init__(
        self,
        diameter: float,
        insulator_thickness: float,
        path: Union[Wire, Edge],
        electrode_material: Material = COPPER,
        insulator_material: Material = QUARTZ,
    ):
        # Make sure path is a list of edges
        if isinstance(path, Wire):
            path = path.edges()
        elif isinstance(path, Edge):
            path = [path]
        elif isinstance(path, Curve):
            path = list(path)
        else:
            raise ValueError("Invalid path type")

        # cable core
        cable_core = []
        for p in path:
            l = Plane(origin=p @ 0, z_dir=p % 0)
            circle = l * Circle(diameter / 2)
            cable_core.append(sweep(circle, p))
        cable_core = Part() + cable_core
        cable_core.label = "cable_core"
        cable_core.material = electrode_material
        cable_core.color = Color(electrode_material.color)

        # cable insulator
        cable_insulator = []
        for p in path:
            l = Plane(origin=p @ 0, z_dir=p % 0)
            circle = l * (
                Circle((diameter / 2) + insulator_thickness) - Circle(diameter / 2)
            )
            cable_insulator.append(sweep(circle, p))
        cable_insulator = Part() + cable_insulator
        cable_insulator.label = "cable_insulator"
        cable_insulator.material = insulator_material
        cable_insulator.color = Color(insulator_material.color)

        # Call super constructor
        super().__init__(
            label="cable",
            children=[
                cable_core,
                cable_insulator,
            ],
        )

        # Make electrode joint
        RigidJoint(
            label="front_connector",
            to_part=self,
            joint_location=Location(Plane(path[0] @ 0, z_dir=-(path[0] % 0))),
        )

        # Make anode cable joint
        RigidJoint(
            label="back_connector", to_part=self, joint_location=path[-1].location_at(1)
        )
