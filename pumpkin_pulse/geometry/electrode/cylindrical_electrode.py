from __future__ import annotations

from typing import Literal, Union

from build123d import *
from build123d import tuplify

from pumpkin_pulse.geometry.electrode.electrode import Electrode


class CylindricalElectrode(Electrode):
    """
    A Cylinderical electrode with a fillet at the top.
    """

    def __init__(
        self,
        outer_diameter: float,
        inner_diameter: float,
        height: float,
        fillet_radius: float = 0,
        rotation: RotationLike = (0, 0, 0),
        align: Union[None, Align, tuple[Align, Align, Align]] = None,
        mode: Mode = Mode.ADD,
    ):
        context: BuildPart = BuildPart._get_context()

        # Create the electrode
        electrode = Circle(outer_diameter / 2)
        if inner_diameter > 0:
            electrode = electrode - Circle(inner_diameter / 2)
        electrode = extrude(electrode, amount=height)

        # Create the fillet
        electrode = fillet(electrode.edges().group_by(Axis.Z)[-1], radius=fillet_radius)

        super().__init__(
            part=electrode, rotation=rotation, align=tuplify(align, 3), mode=mode
        )
