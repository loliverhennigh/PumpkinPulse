
from build123d import *


class Reactor(Co


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
