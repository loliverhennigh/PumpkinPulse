from __future__ import annotations

from typing import Literal, Union

from build123d import *
from build123d import tuplify

class CylindricElectrode(BasePartObject):
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
        electrode= Circle(outer_diameter/2)
        if inner_diameter > 0:
            electrode = electrode - Circle(inner_diameter/2)
        electrode = extrude(electrode, amount=height)

        # Create the fillet
        electrode = fillet(electrode.edges().group_by(Axis.Z)[-1], radius=fillet_radius)

        super().__init__(
            part=electrode, rotation=rotation, align=tuplify(align, 3), mode=mode
        )

        # Make joint to base
        RigidJoint(
            label="base",
            to_part=self,
            joint_location=Location((0, 0, 0)),
        )

class SpokedElectrode(BasePartObject):
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

        # Make joint to base
        RigidJoint(
            label="base",
            to_part=self,
            joint_location=Location((0, 0, 0)),
        )

class AnodeBase(BasePartObject):
    """
    Base for the cathode
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
        base = Circle(outer_diameter/2)
        if inner_diameter > 0:
            base = base - Circle(inner_diameter/2)
        base = extrude(base, amount=height)

        # Make joint to electrode
        super().__init__(
            part=base, rotation=rotation, align=tuplify(align, 3), mode=mode
        )

        # Make joint to electrode
        RigidJoint(
            label="electrode",
            to_part=self,
            joint_location=Location((0, 0, height)),
        )


class CathodeBase(BasePartObject):
    """
    Base for the cathode
    """

    def __init__(
        self,
        plate_outer_diameter: float,
        plate_inner_diameter: float,
        plate_thickness: float,
        base_outer_diameter: float,
        base_inner_diameter: float,
        base_height: float,
        rotation: RotationLike = (0, 0, 0),
        align: Union[None, Align, tuple[Align, Align, Align]] = None,
        mode: Mode = Mode.ADD,
    ):
        context: BuildPart = BuildPart._get_context()

        # Create the electrode
        plate = Circle(plate_outer_diameter/2)
        plate = plate - Circle(plate_inner_diameter/2)
        plate = extrude(plate, amount=plate_thickness)

        # Create the base
        base = Circle(base_outer_diameter/2)
        base = base - Circle(base_inner_diameter/2)
        base = extrude(base, amount=base_height)
        base = Location((0, 0, -base_height)) * base

        # Join the plate and base
        cathode_base = plate + base

        # Make joint to electrode
        super().__init__(
            part=cathode_base, rotation=rotation, align=tuplify(align, 3), mode=mode
        )

        # Make joint to electrode
        RigidJoint(
            label="electrode",
            to_part=self,
            joint_location=Location((0, 0, plate_thickness)),
        )



class AnodeInsulator(BasePartObject):
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
        base = Circle(outer_diameter/2)
        base = base - Circle(inner_diameter/2)
        base = extrude(base, amount=height)

        # Make joint to electrode
        super().__init__(
            part=base, rotation=rotation, align=tuplify(align, 3), mode=mode
        )

        # Make joint to electrode
        RigidJoint(
            label="base",
            to_part=self,
            joint_location=Location((0, 0, 0)),
        )


class InnerBaseInsulator(BasePartObject):
    """
    Insulator sleeve between anode and cathode in the base
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
        base = Circle(outer_diameter/2)
        base = base - Circle(inner_diameter/2)
        base = extrude(base, amount=height)

        # Make joint to electrode
        super().__init__(
            part=base, rotation=rotation, align=tuplify(align, 3), mode=mode
        )

        # Make joint to electrode
        RigidJoint(
            label="electrode",
            to_part=self,
            joint_location=Location((0, 0, height)),
        )

class OuterBaseInsulator(BasePartObject):
    """
    Insulator sleeve for outer cathode in the base
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
        base = Circle(outer_diameter/2)
        base = base - Circle(inner_diameter/2)
        base = extrude(base, amount=height)

        # Make joint to electrode
        super().__init__(
            part=base, rotation=rotation, align=tuplify(align, 3), mode=mode
        )

        # Make joint to electrode
        RigidJoint(
            label="electrode",
            to_part=self,
            joint_location=Location((0, 0, height)),
        )




if __name__ == "__main__":
    from ocp_vscode import *

    # Create electrode 
    anode_electrode = CylindricElectrode(outer_diameter=10, height=15, inner_diameter=5, fillet_radius=1)
    anode_electrode.label = "anode"
    cathode_electrode = SpokedElectrode(outer_diameter=25, inner_diameter=18, height=30, nr_spokes=8, fillet_radius=0.5)
    cathode_electrode.label = "cathode"
    cathode_base = CathodeBase(plate_outer_diameter=25, plate_inner_diameter=18, plate_thickness=3, base_outer_diameter=20, base_inner_diameter=18, base_height=10)
    cathode_base.label = "cathode_base"
    anode_base = AnodeBase(outer_diameter=10, inner_diameter=0, height=13)
    anode_base.label = "cathode_base"
    anode_insulator = AnodeInsulator(outer_diameter=12, inner_diameter=10, height=3)
    anode_insulator.label = "anode_insulator"
    base_insulator = BaseInsulator(outer_diameter=18, inner_diameter=10, height=13)
 
    cathode_electrode.joints["base"].connect_to(cathode_base.joints["electrode"])
    anode_electrode.joints["base"].connect_to(anode_base.joints["electrode"])
    anode_electrode.joints["base"].connect_to(anode_insulator.joints["base"])
    anode_electrode.joints["base"].connect_to(base_insulator.joints["electrode"])

    # Show electrode 
    if "show_object" in locals():
        #show_object(electrode.wrapped, name="pipe")
        #show_object(electrode_assembly.wrapped, name="electrodes")
        #show_object(cathode_electrode, name="cathode", show_joints=True)
        show(cathode_electrode, cathode_base, anode_electrode, anode_base, anode_insulator, base_insulator, render_joints=True)
        #show_object(cathode_base.wrapped, name="cathode_base")
