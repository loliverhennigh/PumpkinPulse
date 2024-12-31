from __future__ import annotations

from typing import Literal, Union
import copy

from build123d import *
from build123d import tuplify

from pumpkin_pulse.geometry.capacitor.capacitor import Capacitor
from pumpkin_pulse.geometry.breakdown.breakdown import Breakdown
from pumpkin_pulse.geometry.resistor.resistor import Resistor
from pumpkin_pulse.geometry.cable.cable import Cable
from pumpkin_pulse.material import Material, QUARTZ, COPPER, VACUUM


class BreakdownCircuit(Compound):
    """
    A capacitor discharge is a compound of a capacitor, a resistor, and a cable.
    """

    def __init__(
        self,
        electrode_material: Material = COPPER,
        insulator_material: Material = QUARTZ,
        loop_outer_diameter: float = 0.5,
        capacitor_width: float = 0.2,
        conductor_plate_thickness: float = 0.05,
        dielectric_thickness: float = 0.1,
        capacitance: float = 1.0,
        capacitor_dielectric_name: str = "dielectric",
        cable_diameter: float = 0.025,
        cable_insulator_thickness: float = 0.025,
        resistor_length: float = 0.1,
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

        # Make breakdown element
        breakdown = Breakdown(
            breakdown_width=capacitor_width,
            breakdown_plate_thickness=conductor_plate_thickness,
            breakdown_thickness=dielectric_thickness,
            cable_diameter=cable_diameter,
            electrode_material=electrode_material,
        )

        # Make cable from resistor to capacitor
        linear_path_length = max(
            (capacitor_width / 2) - resistor_length, resistor_length / 10.0
        )
        linear_path = Line((0, 0, 0), (0, 0, linear_path_length))
        capacitor_thickness = 4 * conductor_plate_thickness + dielectric_thickness
        radius = (
            loop_outer_diameter / 2 - cable_diameter / 2
        ) - capacitor_thickness / 2
        print(f"radius: {radius}")
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

        # Make cable from resistor to breakdown
        arc_path = CenterArc(
            (0, 0, 0),
            radius=radius,
            start_angle=180,
            arc_size=-90,
        )
        path = (
            linear_path
            + Location((radius, 0, linear_path_length), (90, 0, 0)) * arc_path
        )
        breakdown_anode_cable = Cable(
            path=path,
            diameter=cable_diameter,
            insulator_thickness=cable_insulator_thickness,
        )
        breakdown_cathode_cable = copy.deepcopy(breakdown_anode_cable)

        # Connect capacitor to cable
        capacitor.joints["front_connector"].connect_to(
            anode_cable.joints["back_connector"]
        )
        capacitor.joints["back_connector"].connect_to(
            cathode_cable.joints["back_connector"]
        )

        # Connect cable to resistor
        anode_cable.joints["front_connector"].connect_to(
            anode_resistor.joints["front_connector"]
        )
        cathode_cable.joints["front_connector"].connect_to(
            cathode_resistor.joints["front_connector"]
        )

        # Connect breakdown cable to resistor
        anode_resistor.joints["back_connector"].connect_to(
            breakdown_anode_cable.joints["front_connector"]
        )
        cathode_resistor.joints["back_connector"].connect_to(
            breakdown_cathode_cable.joints["front_connector"]
        )

        # Connect cable to breakdown
        breakdown_anode_cable.joints["back_connector"].connect_to(
            breakdown.joints["back_connector"]
        )

        # Create reactor
        super().__init__(
            label="reactor",
            children=[
                capacitor,
                anode_resistor,
                cathode_resistor,
                breakdown,
                anode_cable,
                cathode_cable,
                breakdown_anode_cable,
                breakdown_cathode_cable,
            ],
        )


if __name__ == "__main__":
    from ocp_vscode import *

    breakdown_circuit = BreakdownCircuit()
    show(breakdown_circuit, render_joints=True)
