import warp as wp

from pumpkin_pulse.struct.field import Fielduint8, Fieldfloat32

@wp.struct
class MaterialProperties:
    # Id information
    id: Fielduint8
    mc_id: Fielduint8 # Marching cube id, this determins the triangles in the mesh

    # Material information mappings
    # Electrical properties
    eps_mapping: wp.array(dtype=wp.float32)
    mu_mapping: wp.array(dtype=wp.float32)
    sigma_mapping: wp.array(dtype=wp.float32)

    # Thermal properties
    specific_heat_mapping: wp.array(dtype=wp.float32)
    density_mapping: wp.array(dtype=wp.float32)
    thermal_conductivity_mapping: wp.array(dtype=wp.float32)

    # Particle properties
    solid_fraction_mapping: wp.array(dtype=wp.uint8) # 0: vacuum, 1: solid
    solid_type_mapping: wp.array(dtype=wp.uint8) # 0: reflective, 1: absorptive, 2: stop, 3: thermalize
