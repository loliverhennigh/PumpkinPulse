from __future__ import annotations

from typing import Literal, Union
import copy

from build123d import *
from build123d import tuplify

from pumpkin_pulse.geometry.electrode.cylindrical_electrode import (
    CylindricalElectrode,
)
from pumpkin_pulse.geometry.electrode.llp_electrode import LLPElectrode
from pumpkin_pulse.geometry.feedthrough.feedthrough import Feedthrough
from pumpkin_pulse.geometry.capacitor.capacitor import Capacitor
from pumpkin_pulse.geometry.resistor.resistor import Resistor
from pumpkin_pulse.geometry.cable.cable import Cable
from pumpkin_pulse.geometry.chamber.chamber import Chamber
from pumpkin_pulse.material import Material, QUARTZ, COPPER, VACUUM


class LLPReactor(Compound):
    """
    LLP reactor
    A reactor is a compound of electrodes, feedthroughs, capacitors, resistors, and cables.

    TODO: Refactor so that the reactor is a base class and the LLPReactor is a subclass.
    """

    def __init__(
        self,
        electrode_material: Material = COPPER,
        insulator_material: Material = QUARTZ,
        anode_height: float = 15,
        anode_outer_diameter: float = 10,
        anode_inner_diameter: float = 5,
        anode_fillet_radius: float = 1,
        insulator_thickness: float = 1,
        insulator_height: float = 5,
        cathode_height: float = 30,
        cathode_outer_diameter: float = 25,
        cathode_inner_diameter: float = 18,
        cathode_nr_spokes: int = 6,
        cathode_spoke_pitch: float = 200.0,
        cathode_fillet_radius: float = 0.5,
        capacitor_width: float = 30,
        conductor_plate_thickness: float = 2,
        dielectric_thickness: float = 2,
        capacitance: float = 1.0,
        cable_diameter: float = 2.5,
        cable_insulator_thickness: float = 0.5,
        feedthrough_length: float = 10,
        resistor_length: float = 5,
        resistance: float = 1.0,
        chamber_diameter: float = 40,
        chamber_height: float = 50,
        chamber_thickness: float = 1.5,
    ):
        # Create electrode
        llp_electrode = LLPElectrode(
            anode_height=anode_height,
            anode_outer_diameter=anode_outer_diameter,
            anode_inner_diameter=anode_inner_diameter,
            anode_fillet_radius=anode_fillet_radius,
            insulator_thickness=insulator_thickness,
            insulator_height=insulator_height,
            cathode_height=cathode_height,
            cathode_outer_diameter=cathode_outer_diameter,
            cathode_inner_diameter=cathode_inner_diameter,
            cathode_nr_spokes=cathode_nr_spokes,
            cathode_spoke_pitch=cathode_spoke_pitch,
            cathode_fillet_radius=cathode_fillet_radius,
        )

        # Create feedthrough
        feedthrough = Feedthrough(
            anode_outer_diameter=anode_outer_diameter,
            cathode_inner_diameter=cathode_inner_diameter,
            cathode_outer_diameter=cathode_outer_diameter,
            insulator_thickness=insulator_thickness,
            feedthrough_length=feedthrough_length,
        )

        # Make capacitor
        capacitor = Capacitor(
            capacitor_width=capacitor_width,
            conductor_plate_thickness=conductor_plate_thickness,
            dielectric_thickness=dielectric_thickness,
            capacitance=capacitance,
            cable_diameter=cable_diameter,
            cable_insulator_thickness=cable_insulator_thickness,
            electrode_material=electrode_material,
            insulator_material=insulator_material,
        )

        # Resistors for anode and cathode
        anode_resistor = Resistor(
            diameter=cable_diameter,
            insulator_thickness=cable_insulator_thickness,
            length=resistor_length,
            resistance=resistance,
            insulator_material=insulator_material,
        )
        cathode_resistor = Resistor(
            diameter=cable_diameter,
            insulator_thickness=cable_insulator_thickness,
            length=resistor_length,
            resistance=resistance,
            insulator_material=insulator_material,
        )

        # Make cable from resistor to capacitor
        linear_path_length = max((capacitor_width / 2) - resistor_length, 1.0)
        linear_path = Line((0, 0, 0), (0, 0, linear_path_length))
        capacitor_thickness = 4 * conductor_plate_thickness + dielectric_thickness
        radius = (
            cathode_outer_diameter / 2 - cable_diameter / 2
        ) - capacitor_thickness / 2
        arc_path = CenterArc(
            (0, 0, 0),
            radius=radius,
            start_angle=0,
            arc_size=90,
        )
        path = (
            linear_path
            + Location((-radius, 0, linear_path_length), (90, 0, 0)) * arc_path
        )
        anode_cable = Cable(
            path=path,
            diameter=cable_diameter,
            insulator_thickness=cable_insulator_thickness,
        )
        cathode_cable = copy.deepcopy(anode_cable)

        # Make chamber
        chamber = Chamber(
            feedthrough_diameter=cathode_outer_diameter + 2.0 * insulator_thickness,
            chamber_diameter=chamber_diameter,
            chamber_height=chamber_height,
            chamber_thickness=chamber_thickness,
            material=electrode_material,
        )

        # Make vacuum area
        vacuum_area = Circle(chamber_diameter / 2)
        vacuum_area = extrude(vacuum_area, amount=chamber_height)
        vacuum_area = vacuum_area - [child for child in chamber.children]
        vacuum_area = vacuum_area - [child for child in llp_electrode.children]
        vacuum_area.label = "vacuum_area"
        vacuum_area.color = Color(VACUUM.color)
        vacuum_area.material = VACUUM

        # Connect feedthrough to electrode
        llp_electrode.joints["feedthrough"].connect_to(feedthrough.joints["electrode"])

        # Connect chamber to electrode
        llp_electrode.joints["feedthrough"].connect_to(chamber.joints["electrode"])

        # Connect resistors to feedthrough
        feedthrough.joints["anode_cable"].connect_to(
            anode_resistor.joints["front_connector"]
        )
        feedthrough.joints["cathode_cable"].connect_to(
            cathode_resistor.joints["front_connector"]
        )

        # Connect cable to resistor
        anode_resistor.joints["back_connector"].connect_to(
            anode_cable.joints["front_connector"]
        )
        cathode_resistor.joints["back_connector"].connect_to(
            cathode_cable.joints["front_connector"]
        )

        # Connect capacitor to cable
        anode_cable.joints["back_connector"].connect_to(
            capacitor.joints["front_connector"]
        )

        # Create reactor
        super().__init__(
            label="reactor",
            children=[
                llp_electrode,
                feedthrough,
                capacitor,
                anode_resistor,
                cathode_resistor,
                anode_cable,
                cathode_cable,
                chamber,
                vacuum_area,
            ],
        )


if __name__ == "__main__":
    from ocp_vscode import *

    reactor = LLPReactor()
    show(reactor, render_joints=True)
