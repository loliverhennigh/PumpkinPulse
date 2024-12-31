from __future__ import annotations

from typing import Literal, Union

from build123d import *
from build123d import tuplify

from pumpkin_pulse.geometry.electrode.spoked_electrode import SpokedElectrode
from pumpkin_pulse.geometry.electrode.anode_insulator import AnodeInsulator
from pumpkin_pulse.material import Material, QUARTZ, COPPER
from pumpkin_pulse.geometry.electrode.cylindrical_electrode import (
    CylindricalElectrode,
)


class LLPElectrode(Compound):
    """
    Base class for electrode from LLP Fusion patent.
    """

    def __init__(
        self,
        anode_height: float,
        anode_outer_diameter: float,
        anode_inner_diameter: float,
        anode_fillet_radius: float,
        insulator_thickness: float,
        insulator_height: float,
        cathode_height: float,
        cathode_outer_diameter: float,
        cathode_inner_diameter: float,
        cathode_nr_spokes: int,
        cathode_spoke_pitch: float,
        cathode_fillet_radius: float,
        electrode_material: Material = COPPER,
        insulator_material: Material = QUARTZ,
    ):
        # Create anode
        anode_electrode = CylindricalElectrode(
            outer_diameter=anode_outer_diameter,
            height=anode_height,
            inner_diameter=anode_inner_diameter,
            fillet_radius=anode_fillet_radius,
        )
        anode_electrode.label = "anode"
        anode_electrode.material = electrode_material
        anode_electrode.color = Color(electrode_material.color)
        #anode_insulator = AnodeInsulator(
        #    outer_diameter=anode_outer_diameter + insulator_thickness,
        #    inner_diameter=anode_outer_diameter,
        #    height=insulator_height,
        #)
        #anode_insulator.label = "anode_insulator"
        #anode_insulator.material = insulator_material
        #anode_insulator.color = Color(insulator_material.color)
        #anode_electrode.joints["feedthrough"].connect_to(
        #    anode_insulator.joints["feedthrough"]
        #)

        # Create cathode
        cathode_electrode = SpokedElectrode(
            outer_diameter=cathode_outer_diameter,
            inner_diameter=cathode_inner_diameter,
            height=cathode_height,
            nr_spokes=cathode_nr_spokes,
            pitch=cathode_spoke_pitch,
            fillet_radius=cathode_fillet_radius,
        )
        cathode_electrode.label = "cathode"
        cathode_electrode.material = electrode_material
        cathode_electrode.color = Color(electrode_material.color)
        #cathode_electrode.joints["feedthrough"].connect_to(
        #    anode_insulator.joints["feedthrough"]
        #)

        # Call super constructor
        super().__init__(
            label="LLP_Electrode",
            children=[
                anode_electrode,
                #anode_insulator,
                cathode_electrode,
            ],
        )

        # Make joints
        RigidJoint(
            label="feedthrough",
            to_part=self,
            joint_location=Location((0, 0, 0)),
        )


if __name__ == "__main__":
    from ocp_vscode import *

    # Create llp electrode
    llp_electrode = LLPElectrode(
        anode_height=15,
        anode_outer_diameter=10,
        anode_inner_diameter=5,
        anode_fillet_radius=1,
        cathode_height=30,
        cathode_outer_diameter=25,
        cathode_inner_diameter=18,
        cathode_nr_spokes=8,
        cathode_spoke_pitch=0.5,
        cathode_fillet_radius=0.5,
    )

    # Show electrode
    colors = ColorMap.listed(
        colors=[part.material.color for part in llp_electrode.children]
    )
    show(*llp_electrode.children, colors=colors)
