from __future__ import annotations

from typing import Literal, Union

from build123d import *
from build123d import tuplify

from pumpkin_pulse.material import Material, QUARTZ, COPPER
from pumpkin_pulse.geometry.cable.cable import Cable


class Coil(Compound):
    """
    Cable for a given curve
    """

    def __init__(
        self,
        diameter: float,
        cable_diameter: float,
        insulator_thickness: float,
        nr_turns_z: int,
        nr_turns_r: int,
        electrode_material: Material = COPPER,
        insulator_material: Material = QUARTZ,
    ):

        # Calculate parameters
        outer_radius = diameter / 2
        pitch = 1.25 * (cable_diameter + 2 * insulator_thickness)
        helix_height = pitch * nr_turns_z

        # Set dummy path
        path = None

        # Create helix path
        for i in range(nr_turns_r):

            # Calculate radius for this layer
            radius = outer_radius + i * pitch

            # Create helical path
            helix_path = Helix(
                pitch=pitch,
                height=helix_height,
                radius=radius,
                #direction=(0, 0, 1 if i % 2 == 0 else -1),
                lefthand=False if i % 2 == 0 else True,
            )
            if path is not None:
                end_point = path @ 1
                start_point = helix_path @ 1
                translation = end_point - start_point
                helix_path = Location(translation) * helix_path

            # Create first half arc path
            first_half_arc_path = CenterArc(
                (0, 0, 0),
                radius=radius + pitch / 2,
                start_angle=0,
                arc_size=180,
            )
            end_point = helix_path @ 1
            start_point = first_half_arc_path @ 0
            translation = end_point - start_point
            first_half_arc_path = Location(translation) * first_half_arc_path

            # Create second half arc path
            second_half_arc_path = CenterArc(
                (0, 0, 0),
                radius=radius + pitch,
                start_angle=180,
                arc_size=180,
            )
            end_point = first_half_arc_path @ 1
            start_point = second_half_arc_path @ 0
            translation = end_point - start_point
            second_half_arc_path = Location(translation) * second_half_arc_path

            if i == 0:
                path = helix_path + first_half_arc_path + second_half_arc_path
            elif i < nr_turns_r - 1:
                path = path + helix_path + first_half_arc_path + second_half_arc_path
            else:
                pass
                #path = path + helix_path

        #path = spiral_path
        #path = helix_path
        #for i in range(nr_turns_r):
        #    # Calculate radius for this layer
        #    radius = innermost_radius + i * radius_step
        #    # Create helical path
        #    # Create cable

        ## Make path
        #linear_path = Line(
        #    (0, 0, 0), (0, 0, 1)
        #)

        #path = linear_path + ...
 
        # Make cable
        cable = Cable(
            path=path,
            diameter=cable_diameter,
            insulator_thickness=insulator_thickness,
            electrode_material=electrode_material,
            insulator_material=insulator_material,
        )

        # Call super constructor
        super().__init__(
            label="coil",
            children=[
                cable,
                path,
            ],
        )

        ## Make electrode joint
        #RigidJoint(
        #    label="front_connector",
        #    to_part=self,
        #    joint_location=Location(Plane(path[0] @ 0, z_dir=-(path[0] % 0))),
        #)

        ## Make anode cable joint
        #RigidJoint(
        #    label="back_connector", to_part=self, joint_location=path[-1].location_at(1)
        #)


if __name__ == "__main__":
    from ocp_vscode import *

    # Make a coil
    coil = Coil(
        diameter=0.25,
        cable_diameter=0.005,
        insulator_thickness=0.002,
        nr_turns_z=3,
        nr_turns_r=3,
    )
    print(coil)
    coil.export_stl("spiral.stl")
    print("STL exported")
    #show(coil, render_joints=True)
