from __future__ import annotations

from typing import Literal, Union
import copy

from build123d import *
from build123d import tuplify

from dense_plasma_focus.geometry.capacitor.capacitor import Capacitor
from dense_plasma_focus.geometry.resistor.resistor import Resistor
from dense_plasma_focus.geometry.cable.cable import Cable
from dense_plasma_focus.material import Material, QUARTZ, COPPER, VACUUM


class CapacitorDischarge(Compound):
    """
    A capacitor discharge is a compound of a capacitor, a resistor, and a cable.
    TODO: Refactor so that the reactor is a base class and the LLPReactor is a subclass.
    """

    def __init__(
        self,
        electrode_material: Material = COPPER,
        insulator_material: Material = QUARTZ,
        loop_outer_diameter: float = 30,
        capacitor_width: float = 30,
        conductor_plate_thickness: float = 2,
        dielectric_thickness: float = 2,
        capacitance: float = 1.0,
        capacitor_dielectric_name: str = "dielectric",
        cable_diameter: float = 2.5,
        cable_insulator_thickness: float = 1.0,
        resistor_length: float = 5,
        anode_resistance: float = 1.0,
        cathode_resistance: float = 1.0,
        anode_resistor_name: str = "anode_resistor",
        cathode_resistor_name: str = "cathode_resistor",
    ):

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
            dielectric_material_name=capacitor_dielectric_name,
        )
    
        # Resistors for anode and cathode
        anode_resistor = Resistor(
            diameter=cable_diameter,
            insulator_thickness=cable_insulator_thickness,
            length=resistor_length,
            resistance=anode_resistance,
            insulator_material=insulator_material,
            resistive_material_name=anode_resistor_name,
        )
        cathode_resistor = Resistor(
            diameter=cable_diameter,
            insulator_thickness=cable_insulator_thickness,
            length=resistor_length,
            resistance=cathode_resistance,
            insulator_material=insulator_material,
            resistive_material_name=cathode_resistor_name,
        )
    
        # Make cable from resistor to capacitor
        linear_path_length = max((capacitor_width/2) - resistor_length, 1.0)
        linear_path = Line((0, 0, 0), (0, 0, linear_path_length))
        capacitor_thickness = 4 * conductor_plate_thickness + dielectric_thickness
        radius = (loop_outer_diameter/2 - cable_diameter/2) - capacitor_thickness/2
        arc_path = CenterArc(
            (0, 0, 0),
            radius=radius,
            start_angle=0,
            arc_size=90,
        )
        path = linear_path + Location((-radius, 0, linear_path_length), (90, 0, 0)) * arc_path
        anode_cable = Cable(
            path=path,
            diameter=cable_diameter,
            insulator_thickness=cable_insulator_thickness,
        )
        cathode_cable = copy.deepcopy(anode_cable)

        # Make cable from anode resistor to cathode resistor
        arc_path_1 = CenterArc(
            (0, 0, 0),
            radius=radius,
            start_angle=90,
            arc_size=90,
        )
        linear_path = Line((0, 0, 0), (0, 0, 4 * conductor_plate_thickness + dielectric_thickness))
        linear_path = Location((-radius, 0, 0), (90, 0, 0)) * linear_path
        arc_path_2 = CenterArc(
            (0, 0, 0),
            radius=radius,
            start_angle=270,
            arc_size=90,
        )
        arc_path_2 = Location((0, -(4 * conductor_plate_thickness + dielectric_thickness), 0), (0, 180, 0)) * arc_path_2
        path = arc_path_1 + linear_path + arc_path_2
        cable = Cable(
            path=path,
            diameter=cable_diameter,
            insulator_thickness=cable_insulator_thickness,
        )
        RigidJoint(
            label="front_connector",
            to_part=cable,
            joint_location=Location((0, radius, 0), (0, 90, 90)),
        )

        # Connect capacitor to cable
        capacitor.joints["front_connector"].connect_to(anode_cable.joints["back_connector"])
        capacitor.joints["back_connector"].connect_to(cathode_cable.joints["back_connector"])

        # Connect cable to resistor
        anode_cable.joints["front_connector"].connect_to(anode_resistor.joints["front_connector"])
        cathode_cable.joints["front_connector"].connect_to(cathode_resistor.joints["front_connector"])

        # Connect cable to resistor
        anode_resistor.joints["back_connector"].connect_to(cable.joints["front_connector"])

        # Create reactor
        super().__init__(
            label="reactor",
            children=[
                capacitor,
                anode_resistor,
                cathode_resistor,
                anode_cable,
                cathode_cable,
                cable,
            ],
        )


if __name__ == "__main__":
    from ocp_vscode import *

    capacitor_discharge = CapacitorDischarge()
    show(capacitor_discharge, render_joints=True)
