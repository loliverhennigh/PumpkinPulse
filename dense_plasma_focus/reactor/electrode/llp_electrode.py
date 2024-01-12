from __future__ import annotations

from typing import Literal, Union

from build123d import *
from build123d import tuplify

from dense_plasma_focus.reactor.electrode.cylindrical_electrode import CylindricalElectrode
from dense_plasma_focus.reactor.electrode.spoked_electrode import SpokedElectrode
from dense_plasma_focus.reactor.electrode.anode_insulator import AnodeInsulator
from dense_plasma_focus.material import Material, QUARTZ, COPPER


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
        anode_electrode = CylindricalElectrode(outer_diameter=10, height=15, inner_diameter=5, fillet_radius=1)
        anode_electrode.label = "anode"
        anode_electrode.material = electrode_material
        anode_insulator = AnodeInsulator(outer_diameter=12, inner_diameter=10, height=3)
        anode_insulator.label = "anode_insulator"
        anode_insulator.material = insulator_material
        print(electrode_material.color)
        anode_electrode.joints["feedthrough"].connect_to(anode_insulator.joints["feedthrough"])

        # Create cathode
        cathode_electrode = SpokedElectrode(outer_diameter=25, inner_diameter=18, height=30, nr_spokes=8, fillet_radius=0.5)
        cathode_electrode.label = "cathode"
        cathode_electrode.material = electrode_material
        cathode_electrode.joints["feedthrough"].connect_to(anode_insulator.joints["feedthrough"])

        # Call super constructor
        super().__init__(
            label="LLP_Electrode",
            children=[
                anode_electrode,
                anode_insulator,
                cathode_electrode,
            ],
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
    colors = ColorMap.listed(colors=[part.material.color for part in llp_electrode.children])
    show(*llp_electrode.children, colors=colors)
