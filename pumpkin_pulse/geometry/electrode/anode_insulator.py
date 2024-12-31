#from __future__ import annotations

from typing import Literal, Union

from build123d import *
from build123d import tuplify

from pumpkin_pulse.geometry.electrode.electrode import Electrode


class AnodeInsulator(Electrode):
    """
    Insulator sleeve for the anode
    """

    def __init__(
        self,
        outer_diameter: float,
        inner_diameter: float,
        height: float,
        rotation: RotationLike = (0, 0, 0),
        align: Union[None, Align, tuple[Align, Align, Align]] = None,
        mode: Mode = Mode.ADD,
    ):
        context: BuildPart = BuildPart._get_context()

        # Create the base
        base = Circle(outer_diameter / 2)
        base = base - Circle(inner_diameter / 2)
        base = extrude(base, amount=height)

        # Make joint to electrode
        super().__init__(
            part=base, rotation=rotation, align=tuplify(align, 3), mode=mode
        )
