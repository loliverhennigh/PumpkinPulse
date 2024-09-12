# Implosion of a pumpkin cavity

import os
import numpy as np
import warp as wp
from tqdm import tqdm
import time

wp.init()

from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.data import Fieldfloat32, Fielduint8
from pumpkin_pulse.operator import Operator
from pumpkin_pulse.operator.hydro import (
    PrimitiveToConservative,
    ConservativeToPrimitive,
    GetTimeStep,
    EulerUpdate,
)
from pumpkin_pulse.operator.mhd import (
    AddEMSourceTerms,
    GetCurrentDensity,
)
from pumpkin_pulse.operator.electromagnetism import (
    YeeElectricFieldUpdate,
    YeeMagneticFieldUpdate
)
from pumpkin_pulse.operator.saver import FieldSaver
from pumpkin_pulse.functional.indexing import periodic_indexing, periodic_setting
from pumpkin_pulse.functional.stencil import (
    p7_float32_stencil_type,
    p7_uint8_stencil_type,
    p4_float32_stencil_type,
    p4_uint8_stencil_type,
    faces_float32_type,
)


class BlastInitializer(Operator):

    @wp.kernel
    def _initialize_blast(
        density_i: Fieldfloat32,
        velocity_i: Fieldfloat32,
        pressure_i: Fieldfloat32,
        density_e: Fieldfloat32,
        velocity_e: Fieldfloat32,
        pressure_e: Fieldfloat32,
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
        id_field: Fielduint8,
        proton_mass: wp.float32,
        electron_mass: wp.float32,
        cylinder_number_density: wp.float32,
        background_number_density: wp.float32,
        cylinder_pressure: wp.float32,
        background_pressure: wp.float32,
        cylinder_radius: wp.float32,
        insulator_radius: wp.float32,
        insulator_thickness: wp.float32,
    ):

        # Get index
        i, j, k = wp.tid()

        # Get x, y, z
        x = density_i.origin[0] + wp.float32(i) * density_i.spacing[0] + 0.5 * density_i.spacing[0]
        y = density_i.origin[1] + wp.float32(j) * density_i.spacing[1] + 0.5 * density_i.spacing[1]
        z = density_i.origin[2] + wp.float32(k) * density_i.spacing[2] + 0.5 * density_i.spacing[2]

        # Get if inside cylinder
        distance = wp.sqrt(x**2.0 + y**2.0)

        # Set density
        if distance < cylinder_radius:
            density_i.data[0, i, j, k] = cylinder_number_density * proton_mass
            density_e.data[0, i, j, k] = cylinder_number_density * electron_mass
            pressure_i.data[0, i, j, k] = cylinder_pressure
            pressure_e.data[0, i, j, k] = cylinder_pressure
            id_field.data[0, i, j, k] = wp.uint8(0)
        else:
            density_i.data[0, i, j, k] = background_number_density * proton_mass
            density_e.data[0, i, j, k] = background_number_density * electron_mass
            pressure_i.data[0, i, j, k] = background_pressure
            pressure_e.data[0, i, j, k] = background_pressure
            id_field.data[0, i, j, k] = wp.uint8(1)

        # Set velocity
        velocity_i.data[0, i, j, k] = 0.0
        velocity_i.data[1, i, j, k] = 0.0
        velocity_i.data[2, i, j, k] = 0.0
        velocity_e.data[0, i, j, k] = 0.0
        velocity_e.data[1, i, j, k] = 0.0
        velocity_e.data[2, i, j, k] = 0.0
            
        # Set electric field
        electric_field.data[0, i, j, k] = 0.0
        electric_field.data[1, i, j, k] = 0.0
        electric_field.data[2, i, j, k] = 0.0

        # Set magnetic field
        magnetic_field.data[0, i, j, k] = 0.0
        magnetic_field.data[1, i, j, k] = 0.0
        magnetic_field.data[2, i, j, k] = 0.0

        # Set id field for insulator
        if (distance > insulator_radius) and (distance < insulator_radius + insulator_thickness):
            id_field.data[0, i, j, k] = wp.uint8(2)
        #if distance < cylinder_radius + 0.01:
        #    id_field.data[0, i, j, k] = wp.uint8(0)
        #else:
        #    id_field.data[0, i, j, k] = wp.uint8(1)


    def __call__(
        self,
        density_i: Fieldfloat32,
        velocity_i: Fieldfloat32,
        pressure_i: Fieldfloat32,
        density_e: Fieldfloat32,
        velocity_e: Fieldfloat32,
        pressure_e: Fieldfloat32,
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
        id_field: Fielduint8,
        proton_mass: float,
        electron_mass: float,
        cylinder_number_density: float,
        background_number_density: float,
        cylinder_pressure: float,
        background_pressure: float,
        cylinder_radius: float,
        insulator_radius: float,
        insulator_thickness: float,
    ):
        # Launch kernel
        wp.launch(
            self._initialize_blast,
            inputs=[
                density_i,
                velocity_i,
                pressure_i,
                density_e,
                velocity_e,
                pressure_e,
                electric_field,
                magnetic_field,
                id_field,
                proton_mass,
                electron_mass,
                cylinder_number_density,
                background_number_density,
                cylinder_pressure,
                background_pressure,
                cylinder_radius,
                insulator_radius,
                insulator_thickness
            ],
            dim=density_i.shape,
        )

        return density_i, velocity_i, pressure_i, density_e, velocity_e, pressure_e, electric_field, magnetic_field


class CurrentInjector(Operator):

    @wp.kernel
    def _current_injector(
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
        current_density: Fieldfloat32,
        id_field: Fielduint8,
        eps_mapping: wp.array(dtype=wp.float32),
        sigma_e_mapping: wp.array(dtype=wp.float32),
        radius: wp.float32,
        thickness: wp.float32,
        length: wp.float32,
        current_intensity: wp.float32,
    ):

        # Get index
        i, j, k = wp.tid()

        # Get x, y, z
        x = current_density.origin[0] + wp.float32(i) * current_density.spacing[0] + 0.5 * current_density.spacing[0]
        y = current_density.origin[1] + wp.float32(j) * current_density.spacing[1] + 0.5 * current_density.spacing[1]
        z = current_density.origin[2] + wp.float32(k) * current_density.spacing[2] + 0.5 * current_density.spacing[2]

        # Get distance
        x_distance = wp.abs(x)
        y_distance = wp.abs(y)

        # Set id field
        if x_distance > (radius - thickness/2.0) and x_distance < (radius + thickness/2.0):
            id_field.data[0, i, j, k] = wp.uint8(3)
        if y_distance > (radius - thickness/2.0) and y_distance < (radius + thickness/2.0):
            id_field.data[0, i, j, k] = wp.uint8(3)

        # Index id field
        id_0_0_1 = periodic_indexing(id_field.data, id_field.shape, 0, i - 1, j - 1, k)
        id_0_1_0 = periodic_indexing(id_field.data, id_field.shape, 0, i - 1, j, k - 1)
        id_0_1_1 = periodic_indexing(id_field.data, id_field.shape, 0, i - 1, j, k)
        id_1_0_0 = periodic_indexing(id_field.data, id_field.shape, 0, i, j - 1, k - 1)
        id_1_0_1 = periodic_indexing(id_field.data, id_field.shape, 0, i, j - 1, k)
        id_1_1_0 = periodic_indexing(id_field.data, id_field.shape, 0, i, j, k - 1)
        id_1_1_1 = periodic_indexing(id_field.data, id_field.shape, 0, i, j, k)

        # get properties
        eps, sigma_e = YeeElectricFieldUpdate._get_eps_and_sigma_e(
            id_0_0_1, id_0_1_0, id_0_1_1, id_1_0_0, id_1_0_1, id_1_1_0, id_1_1_1, eps_mapping, sigma_e_mapping, i, j, k
        )

        if (id_0_0_1 == wp.uint8(3) or id_0_1_0 == wp.uint8(3) or id_0_1_1 == wp.uint8(3) or id_1_0_0 == wp.uint8(3) or id_1_0_1 == wp.uint8(3) or id_1_1_0 == wp.uint8(3) or id_1_1_1 == wp.uint8(3)):

            # Get electric field needed for current density
            cur = wp.vec3f(0.0, 0.0, 0.0)
            if x_distance > (radius - thickness/2.0 - electric_field.spacing[0]) and x_distance < (radius + thickness/2.0 + electric_field.spacing[0]):
                if x < 0:
                    cur[1] = -current_intensity
                else:
                    cur[1] = current_intensity
            if y_distance > (radius - thickness/2.0 - electric_field.spacing[0]) and y_distance < (radius + thickness/2.0 + electric_field.spacing[0]):
                if y < 0:
                    cur[0] = current_intensity
                else:
                    cur[0] = -current_intensity

            # Convert to current density to electric field
            e = wp.vec3f(0.0, 0.0, 0.0)
            if sigma_e[0] != 0.0:
                e[0] = cur[0] / sigma_e[0]
            if sigma_e[1] != 0.0:
                e[1] = cur[1] / sigma_e[1]
            if sigma_e[2] != 0.0:
                e[2] = cur[2] / sigma_e[2]

            # Set electric field
            magnetic_field.data[0, i, j, k] = 0.0
            magnetic_field.data[1, i, j, k] = 0.0
            magnetic_field.data[2, i, j, k] = -current_intensity
            #electric_field.data[0, i, j, k] += e[0]
            #electric_field.data[1, i, j, k] += e[1]
            #electric_field.data[2, i, j, k] += e[2]

    def __call__(
        self,
        electric_field: Fieldfloat32,
        magnetic_field: Fieldfloat32,
        current_density: Fieldfloat32,
        id_field: Fielduint8,
        eps_mapping: wp.array(dtype=wp.float32),
        sigma_e_mapping: wp.array(dtype=wp.float32),
        radius: float,
        thickness: float,
        length: float,
        current_intensity: float,
    ):
        # Launch kernel
        wp.launch(
            self._current_injector,
            inputs=[
                electric_field,
                magnetic_field,
                current_density,
                id_field,
                eps_mapping,
                sigma_e_mapping,
                radius,
                thickness,
                length,
                current_intensity,
            ],
            dim=current_density.shape,
        )

        return current_density


class ElectricFieldToChargeDensity(Operator):
   """
   Electric field to charge density operator

   Basically a gaussian surface around the charge density cell
   """

   @wp.kernel
   def _electric_field_to_charge_density(
        electric_field: Fieldfloat32,
        charge_density: Fieldfloat32,
        #id_field: Fielduint8,
        #eps_mapping: wp.array(dtype=wp.float32),
        #spacing: wp.vec3f,
        #nr_ghost_cells: wp.int32,
   ):

       # get index
       i, j, k = wp.tid()

       ## get eps for each direction
       #eps = ElectricFieldUpdate._sample_electric_property(solid_id, eps_mapping, i, j, k)
       #eps_x_u = eps[0]
       #eps_x_d = ElectricFieldUpdate._sample_electric_property(solid_id, eps_mapping, i - 1, j, k)[0]
       #eps_y_u = eps[1]
       #eps_y_d = ElectricFieldUpdate._sample_electric_property(solid_id, eps_mapping, i, j - 1, k)[1]
       #eps_z_u = eps[2]
       #eps_z_d = ElectricFieldUpdate._sample_electric_property(solid_id, eps_mapping, i, j, k - 1)[2]

       # Get electric field for each direction
       e_x_u = periodic_indexing(electric_field.data, electric_field.shape, 0, i, j, k)
       e_x_d = periodic_indexing(electric_field.data, electric_field.shape, 0, i - 1, j, k)
       e_y_u = periodic_indexing(electric_field.data, electric_field.shape, 1, i, j, k)
       e_y_d = periodic_indexing(electric_field.data, electric_field.shape, 1, i, j - 1, k)
       e_z_u = periodic_indexing(electric_field.data, electric_field.shape, 2, i, j, k)
       e_z_d = periodic_indexing(electric_field.data, electric_field.shape, 2, i, j, k - 1)

       # Sum electric field dot normal
       eps = 8.85418782e-12
       charge_density.data[0, i, j, k] = eps * (
              (e_x_u - e_x_d) / electric_field.spacing[0]
              + (e_y_u - e_y_d) / electric_field.spacing[1]
              + (e_z_u - e_z_d) / electric_field.spacing[2]
        )

   def __call__(
       self,
       electric_field: wp.array4d(dtype=wp.float32),
       charge_density: wp.array4d(dtype=wp.float32),
   ):
       # Launch kernel
       wp.launch(
           self._electric_field_to_charge_density,
           inputs=[
               electric_field,
               charge_density,
           ],
           dim=charge_density.shape,
       )

       return charge_density



# Make boundary conditions functions
@wp.func
def apply_boundary_conditions_p7(
    rho_stencil: p7_float32_stencil_type,
    vx_stencil: p7_float32_stencil_type,
    vy_stencil: p7_float32_stencil_type,
    vz_stencil: p7_float32_stencil_type,
    p_stencil: p7_float32_stencil_type,
    id_stencil: p7_uint8_stencil_type,
):

    # Apply boundary conditions
    for i in range(6):

        # stencil index
        st_idx = i + 1

        # Wall
        if id_stencil[st_idx] == wp.uint8(1):

            # Get normal
            if st_idx == 1 or st_idx == 2:
                flip_vx = -1.0
            else:
                flip_vx = 1.0
            if st_idx == 3 or st_idx == 4:
                flip_vy = -1.0
            else:
                flip_vy = 1.0
            if st_idx == 5 or st_idx == 6:
                flip_vz = -1.0
            else:
                flip_vz = 1.0

            # Apply wall boundary condition
            rho_stencil[st_idx] = rho_stencil[0]
            vx_stencil[st_idx] = vx_stencil[0] * flip_vx
            vy_stencil[st_idx] = vy_stencil[0] * flip_vy
            vz_stencil[st_idx] = vz_stencil[0] * flip_vz
            p_stencil[st_idx] = p_stencil[0]

    return rho_stencil, vx_stencil, vy_stencil, vz_stencil, p_stencil

@wp.func
def apply_boundary_conditions_faces(
    rho_faces: faces_float32_type,
    vx_faces: faces_float32_type,
    vy_faces: faces_float32_type,
    vz_faces: faces_float32_type,
    p_faces: faces_float32_type,
    rho_stencil: p4_float32_stencil_type,
    vx_stencil: p4_float32_stencil_type,
    vy_stencil: p4_float32_stencil_type,
    vz_stencil: p4_float32_stencil_type,
    p_stencil: p4_float32_stencil_type,
    id_stencil: p4_uint8_stencil_type,
):

    # Apply boundary conditions
    for i in range(3):

        # Check if on the boundary
        if id_stencil[0] != wp.uint8(0) or id_stencil[i + 1] != wp.uint8(0):

            # Get normal
            if i == 0:
                flip_vx, flip_vy, flip_vz = -1.0, 1.0, 1.0
            if i == 1:
                flip_vx, flip_vy, flip_vz = 1.0, -1.0, 1.0
            if i == 2:
                flip_vx, flip_vy, flip_vz = 1.0, 1.0, -1.0

            # Wall
            if id_stencil[0] == wp.uint8(1):
                rho_faces[2 * i] = rho_stencil[i + 1]
                vx_faces[2 * i] = vx_stencil[i + 1] * flip_vx
                vy_faces[2 * i] = vy_stencil[i + 1] * flip_vy
                vz_faces[2 * i] = vz_stencil[i + 1] * flip_vz
                p_faces[2 * i] = p_stencil[i + 1]
            if id_stencil[i + 1] == wp.uint8(1):
                rho_faces[2 * i + 1] = rho_stencil[0]
                vx_faces[2 * i + 1] = vx_stencil[0] * flip_vx
                vy_faces[2 * i + 1] = vy_stencil[0] * flip_vy
                vz_faces[2 * i + 1] = vz_stencil[0] * flip_vz
                p_faces[2 * i + 1] = p_stencil[0]

    return rho_faces, vx_faces, vy_faces, vz_faces, p_faces


if __name__ == '__main__':

    # Geometry parameters
    dx = 0.00025 # m
    l = 0.040
    origin = (-l / 2.0, -l / 2.0, -l / 2.0)
    spacing = (dx, dx, dx)
    shape = (int(l / dx), int(l / dx), int(l / dx))
    nr_cells = shape[0] * shape[1] * shape[2]

    # Electromagnetic Constants
    elementary_charge = 1.60217662e-19
    epsilon_0 = 8.85418782e-12
    mu_0 = 4.0 * 3.14159 * 1.0e-7
    c = (1.0 / np.sqrt(mu_0 * epsilon_0))
    electron_mass = 9.10938356e-31
    proton_mass = 1.6726219e-27
    boltzmann_constant = 1.38064852e-23

    # Fluid Constants
    gamma = (5.0 / 3.0)

    # Plasma cylinder parameters
    cylinder_number_density = 1.0e19
    background_number_density = 1.0e17
    cylinder_radius = 0.01
    cylinder_temperature = 1.0e0 # K
    cylinder_pressure = cylinder_number_density * boltzmann_constant * cylinder_temperature
    background_pressure = background_number_density * boltzmann_constant * cylinder_temperature
    print(f"Cylinder Pressure: {cylinder_pressure}")
    print(f"Background Pressure: {background_pressure}")

    # Current Injector Parameters
    coil_current_intensity = 1.0e6
    coil_radius = 0.015
    coil_thickness = 0.002
    ramp_time = 1.0e-9

    # Insulater Parameters
    insulator_radius = 0.0125
    insulator_thickness = 0.001

    # Time parameters
    simulation_time = 1e-7
    save_frequency = 1e-11
    courant_factor = 0.4
    em_dt = 0.1 * (1.0 / (c * np.sqrt(3.0 / (dx ** 2.0))))

    # Make output directory
    output_dir = f"output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Make the constructor
    constructor = Constructor()

    # Make the operators
    initialize_blast = BlastInitializer()
    current_injector = CurrentInjector()
    primitive_to_conservative = PrimitiveToConservative()
    conservative_to_primitive = ConservativeToPrimitive()
    get_time_step = GetTimeStep()
    euler_update = EulerUpdate(
        apply_boundary_conditions_p7=apply_boundary_conditions_p7,
        apply_boundary_conditions_faces=apply_boundary_conditions_faces,
    )
    add_em_source_terms = AddEMSourceTerms()
    get_current_density = GetCurrentDensity()
    yee_electric_field_update = YeeElectricFieldUpdate()
    yee_magnetic_field_update = YeeMagneticFieldUpdate()
    electric_field_to_charge_density = ElectricFieldToChargeDensity()
    field_saver = FieldSaver()

    # Make the fields
    density_i = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    velocity_i = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    pressure_i = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    density_e = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    velocity_e = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    pressure_e = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    mass_i = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    momentum_i = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    energy_i = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    mass_e = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    momentum_e = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    energy_e = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    electric_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    magnetic_field = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    current_density = constructor.create_field(
        dtype=wp.float32,
        cardinality=3,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    id_field = constructor.create_field(
        dtype=wp.uint8,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    charge_density = constructor.create_field(
        dtype=wp.float32,
        cardinality=1,
        shape=shape,
        origin=origin,
        spacing=spacing
    )
    eps_mapping = wp.from_numpy(np.array([epsilon_0, epsilon_0, 1000.0 * epsilon_0, epsilon_0], dtype=np.float32), dtype=wp.float32)
    mu_mapping = wp.from_numpy(np.array([mu_0, mu_0, 1000.0 * mu_0, mu_0], dtype=np.float32), dtype=wp.float32)
    sigma_e_mapping = wp.from_numpy(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), dtype=wp.float32)
    sigma_m_mapping = wp.from_numpy(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), dtype=wp.float32)

    # Initialize the fields
    (
        density_i,
        velocity_i,
        pressure_i,
        density_e,
        velocity_e,
        pressure_e,
        electric_field,
        magnetic_field,
    ) = initialize_blast(
        density_i,
        velocity_i,
        pressure_i,
        density_e,
        velocity_e,
        pressure_e,
        electric_field,
        magnetic_field,
        id_field,
        proton_mass,
        electron_mass,
        cylinder_number_density,
        background_number_density,
        cylinder_pressure,
        background_pressure,
        cylinder_radius,
        insulator_radius,
        insulator_thickness,
    )

    # Get conservative variables
    mass_i, momentum_i, energy_i = primitive_to_conservative(
        density_i,
        velocity_i,
        pressure_i,
        mass_i,
        momentum_i,
        energy_i,
        gamma
    )
    mass_e, momentum_e, energy_e = primitive_to_conservative(
        density_e,
        velocity_e,
        pressure_e,
        mass_e,
        momentum_e,
        energy_e,
        gamma
    )

    # Save the fields
    field_saver(
        {
            "density_i": density_i,
            #"velocity_i": velocity_i,
            "pressure_i": pressure_i,
            "density_e": density_e,
            #"velocity_e": velocity_e,
            "pressure_e": pressure_e,
            "electric_field": electric_field,
            "magnetic_field": magnetic_field,
            "current_density": current_density,
        },
        os.path.join(output_dir, "initial_conditions.vtk")
    )

    # Run the simulation
    current_time = 0.0
    save_index = 0
    nr_iterations = 0
    tic = time.time()
    with tqdm(total=simulation_time, desc="Simulation Progress") as pbar:
        while current_time < simulation_time:

            # Get primitive variables
            density_i, velocity_i, pressure_i = conservative_to_primitive(
                density_i,
                velocity_i,
                pressure_i,
                mass_i,
                momentum_i,
                energy_i,
                gamma
            )
            density_e, velocity_e, pressure_e = conservative_to_primitive(
                density_e,
                velocity_e,
                pressure_e,
                mass_e,
                momentum_e,
                energy_e,
                gamma
            )

            # Get the time step
            dt_i = get_time_step(
                density_i,
                velocity_i,
                pressure_i,
                id_field,
                courant_factor,
                gamma,
            )
            dt_e = get_time_step(
                density_e,
                velocity_e,
                pressure_e,
                id_field,
                courant_factor,
                gamma,
            )
            dt = min(dt_i, dt_e, em_dt)
            print(f"Time Step EM: {em_dt}")
            print(f"Time Step Ion: {dt_i}")
            print(f"Time Step Elec: {dt_e}")
            print(f"Time Step: {dt}")

            # Get the current density
            current_density.data.zero_()
            current_density = get_current_density(
                density_i,
                velocity_i,
                pressure_i,
                current_density,
                id_field,
                proton_mass,
                elementary_charge,
            )
            current_density = get_current_density(
                density_e,
                velocity_e,
                pressure_e,
                current_density,
                id_field,
                electron_mass,
                -elementary_charge,
            )
            current_density = current_injector(
                electric_field,
                magnetic_field,
                current_density,
                id_field,
                eps_mapping,
                sigma_e_mapping,
                coil_radius,
                coil_thickness,
                l,
                min(coil_current_intensity * current_time / ramp_time, coil_current_intensity),
            )

            # Update Conserved Variables
            mass_i, momentum_i, energy_i = euler_update(
                density_i,
                velocity_i,
                pressure_i,
                mass_i,
                momentum_i,
                energy_i,
                id_field,
                gamma,
                dt,
            )
            mass_e, momentum_e, energy_e = euler_update(
                density_e,
                velocity_e,
                pressure_e,
                mass_e,
                momentum_e,
                energy_e,
                id_field,
                gamma,
                dt,
            )

            # Update the magnetic field
            magnetic_field = yee_magnetic_field_update(
                electric_field,
                magnetic_field,
                id_field,
                mu_mapping,
                sigma_m_mapping,
                dt
            )

            # Update the electric
            electric_field = yee_electric_field_update(
                electric_field,
                magnetic_field,
                current_density,
                id_field,
                eps_mapping,
                sigma_e_mapping,
                dt
            )

            # Add EM Source Terms
            mass_i, momentum_i, energy_i = add_em_source_terms(
                density_i,
                velocity_i,
                pressure_i,
                electric_field,
                magnetic_field,
                mass_i,
                momentum_i,
                energy_i,
                id_field,
                proton_mass,
                elementary_charge,
                mu_0,
                dt,
            )
            mass_e, momentum_e, energy_e = add_em_source_terms(
                density_e,
                velocity_e,
                pressure_e,
                electric_field,
                magnetic_field,
                mass_e,
                momentum_e,
                energy_e,
                id_field,
                electron_mass,
                -elementary_charge,
                mu_0,
                dt,
            )

            # Get the charge density
            charge_density = electric_field_to_charge_density(
                electric_field,
                charge_density,
            )

            # Check if time passes save frequency
            remander = current_time % save_frequency 
            if (remander + dt) > save_frequency:
                save_index += 1
                print(f"Saved {save_index} files")
                field_saver(
                    {
                        "density_i": density_i,
                        "velocity_i": velocity_i,
                        "momentum_i": momentum_i,
                        "pressure_i": pressure_i,
                        "density_e": density_e,
                        "velocity_e": velocity_e,
                        "momentum_e": momentum_e,
                        "pressure_e": pressure_e,
                        "electric_field": electric_field,
                        "magnetic_field": magnetic_field,
                        "current_density": current_density,
                        "id_field": id_field,
                        "charge_density": charge_density,
                    },
                    os.path.join(output_dir, f"t_{str(save_index).zfill(4)}.vtk")
                )
                #if save_index == 10:
                #    exit()
                # Print total mass
            #print(f"Total Mass: {np.sum(mass_e.data.numpy())}")

            # Compute MUPS
            if nr_iterations % 10 == 0:
                wp.synchronize()
                toc = time.time()
                mups = nr_cells * nr_iterations / (toc - tic) / 1.0e6
                print(f"Iterations: {nr_iterations}")
                print(f"MUPS: {mups}")

            # Update the time
            current_time += dt

            # Update the progress bar
            pbar.update(dt)

            # Update the number of iterations
            nr_iterations += 1
