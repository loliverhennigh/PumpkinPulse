from __future__ import annotations

from typing import Literal, Union

from build123d import *
from build123d import tuplify


class Electrode(BasePartObject):
    """
    Base class for electrodes
    """

    def __init__(
        self,
        part: Part,
        rotation: RotationLike = (0, 0, 0),
        align: Union[None, Align, tuple[Align, Align, Align]] = None,
        mode: Mode = Mode.ADD,
    ):
        context: BuildPart = BuildPart._get_context()

        super().__init__(
            part=part, rotation=rotation, align=tuplify(align, 3), mode=mode
        )

        # Make joint to base
        RigidJoint(
            label="feedthrough",
            to_part=self,
            joint_location=Location((0, 0, 0)),
        )
