from __future__ import annotations

from typing import Literal, Union

from build123d import *
from build123d import tuplify

from dense_plasma_focus.geometry.electrode.electrode import Electrode

class SpokedElectrode(Electrode):
    """
    A Cylinderical Electrode with spokes
    """

    def __init__(
        self,
        outer_diameter: float,
        inner_diameter: float,
        height: float,
        nr_spokes: int,
        pitch: float = 200.0,
        fillet_radius: float = 0.1,
        rotation: RotationLike = (0, 0, 0),
        align: Union[None, Align, tuple[Align, Align, Align]] = None,
        mode: Mode = Mode.ADD,
    ):
        context: BuildPart = BuildPart._get_context()

        # Get radius of electrode center
        polar_radius = (outer_diameter + inner_diameter)/4

        # Create circle for base of electrode
        base = Circle((outer_diameter - inner_diameter)/4)

        # Create the helix path
        path = Helix(pitch=pitch, height=height, radius=outer_diameter/2)
        
        # Create the electrode 
        spokes = [loc * base for loc in PolarLocations(radius=polar_radius, count=nr_spokes)]
        paths = [loc * path for loc in PolarLocations(radius=polar_radius, count=nr_spokes)]

        # Sweep the spokes
        spokes = [sweep(s, path=p) for s, p in zip(spokes, paths)]

        # Create the electrode
        electrode = Part() + spokes

        # Create the fillet
        electrode = fillet(electrode.edges().group_by(Axis.Z)[-1], radius=fillet_radius)

        super().__init__(
            part=electrode, rotation=rotation, align=tuplify(align, 3), mode=mode
        )
