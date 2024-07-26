# Base class for electromagnetism operators

from typing import Union
import warp as wp

from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.compute_backend import ComputeBackend
from pumpkin_pulse.material import Material


class ElectricFieldUpdate(Operator):
    """
    Electric field update operator
    """

    @wp.func
    def _sample_electric_property(
        solid_id: wp.array4d(dtype=wp.uint8),
        material_property: wp.array(dtype=wp.float32),
        i: wp.int32,
        j: wp.int32,
        k: wp.int32,
    ):
        # Get material property
        prop_0_0_1 = material_property[wp.int32(solid_id[0, i - 1, j - 1, k])]
        prop_0_1_0 = material_property[wp.int32(solid_id[0, i - 1, j, k - 1])]
        prop_0_1_1 = material_property[wp.int32(solid_id[0, i - 1, j, k])]
        prop_1_0_0 = material_property[wp.int32(solid_id[0, i, j - 1, k - 1])]
        prop_1_0_1 = material_property[wp.int32(solid_id[0, i, j - 1, k])]
        prop_1_1_0 = material_property[wp.int32(solid_id[0, i, j, k - 1])]
        prop_1_1_1 = material_property[wp.int32(solid_id[0, i, j, k])]

        # Get average property
        prop_x = (prop_1_1_1 + prop_1_1_0 + prop_1_0_1 + prop_1_0_0) / 4.0
        prop_y = (prop_1_1_1 + prop_1_1_0 + prop_0_1_1 + prop_0_1_0) / 4.0
        prop_z = (prop_1_1_1 + prop_1_0_1 + prop_0_1_1 + prop_0_0_1) / 4.0

        return wp.vec3(prop_x, prop_y, prop_z)

    @wp.kernel
    def _update_electric_field(
        electric_field: wp.array4d(dtype=wp.float32),
        magnetic_field: wp.array4d(dtype=wp.float32),
        impressed_current: wp.array4d(dtype=wp.float32),
        solid_id: wp.array4d(dtype=wp.uint8),
        eps_mapping: wp.array(dtype=wp.float32),
        sigma_e_mapping: wp.array(dtype=wp.float32),
        spacing: wp.vec3f,
        dt: wp.float32,
        nr_ghost_cells: wp.int32,
    ):
        # get index
        i, j, k = wp.tid()

        # Skip ghost cells
        i += nr_ghost_cells
        j += nr_ghost_cells
        k += nr_ghost_cells

        # get properties
        eps = ElectricFieldUpdate._sample_electric_property(
            solid_id, eps_mapping, i, j, k
        )
        sigma_e = ElectricFieldUpdate._sample_electric_property(
            solid_id, sigma_e_mapping, i, j, k
        )

        # Get coefficients
        _denom = 2.0 * eps + sigma_e * dt
        c_ee = wp.cw_div(2.0 * eps - sigma_e * dt, _denom)
        c_eh = (2.0 * dt) / wp.cw_mul(spacing, _denom)
        c_ej = (-2.0 * dt) / _denom

        # Get magnetic field stencil
        m_x_1_1_1 = magnetic_field[0, i, j, k]
        m_x_1_0_1 = magnetic_field[0, i, j - 1, k]
        m_x_1_1_0 = magnetic_field[0, i, j, k - 1]
        m_y_1_1_1 = magnetic_field[1, i, j, k]
        m_y_0_1_1 = magnetic_field[1, i - 1, j, k]
        m_y_1_1_0 = magnetic_field[1, i, j, k - 1]
        m_z_1_1_1 = magnetic_field[2, i, j, k]
        m_z_0_1_1 = magnetic_field[2, i - 1, j, k]
        m_z_1_0_1 = magnetic_field[2, i, j - 1, k]

        # Apply zero flux boundary conditions
        #if i == 1:
        #    m_y_0_1_1 = m_y_1_1_1
        #    m_z_0_1_1 = m_z_1_1_1
        #if j == 1:
        #    m_x_1_0_1 = m_x_1_1_1
        #    m_z_1_0_1 = m_z_1_1_1
        #if k == 1:
        #    m_x_1_1_0 = m_x_1_1_1
        #    m_y_1_1_0 = m_y_1_1_1

        # Get curl of magnetic field
        curl_h_x = (m_z_1_1_1 - m_z_1_0_1) - (m_y_1_1_1 - m_y_1_1_0)
        curl_h_y = (m_x_1_1_1 - m_x_1_1_0) - (m_z_1_1_1 - m_z_0_1_1)
        curl_h_z = (m_y_1_1_1 - m_y_0_1_1) - (m_x_1_1_1 - m_x_1_0_1)
        curl_h = wp.vec3(curl_h_x, curl_h_y, curl_h_z)

        # compute new electric field
        e = wp.vec3f(
            electric_field[0, i, j, k],
            electric_field[1, i, j, k],
            electric_field[2, i, j, k],
        )
        cur = wp.vec3f(
            impressed_current[0, i, j, k],
            impressed_current[1, i, j, k],
            impressed_current[2, i, j, k],
        )
        new_e = wp.cw_mul(c_ee, e) + wp.cw_mul(c_eh, curl_h) + wp.cw_mul(c_ej, cur)

        # Set electric field
        electric_field[0, i, j, k] = new_e[0]
        electric_field[1, i, j, k] = new_e[1]
        electric_field[2, i, j, k] = new_e[2]

    def __call__(
        self,
        electric_field: wp.array4d(dtype=wp.float32),
        magnetic_field: wp.array4d(dtype=wp.float32),
        impressed_current: wp.array4d(dtype=wp.float32),
        solid_id: wp.array4d(dtype=wp.uint8),
        eps_mapping: wp.array(dtype=wp.float32),
        sigma_e_mapping: wp.array(dtype=wp.float32),
        spacing: Union[float, tuple[float, float, float]],
        dt: float,
        nr_ghost_cells: int = 1,
    ):
        # Launch kernel
        wp.launch(
            self._update_electric_field,
            inputs=[
                electric_field,
                magnetic_field,
                impressed_current,
                solid_id,
                eps_mapping,
                sigma_e_mapping,
                spacing,
                dt,
                nr_ghost_cells,
            ],
            dim=[x - 2 * nr_ghost_cells for x in solid_id.shape[1:]],
        )

        return electric_field


class MagneticFieldUpdate(Operator):
    """
    Magnetic field update operator
    """

    @wp.func
    def _sample_magnetic_property(
        solid_id: wp.array4d(dtype=wp.uint8),
        material_property: wp.array(dtype=wp.float32),
        i: wp.int32,
        j: wp.int32,
        k: wp.int32,
    ):
        # Get material property
        prop_1_1_1 = material_property[wp.int32(solid_id[0, i, j, k])]
        prop_0_1_1 = material_property[wp.int32(solid_id[0, i - 1, j, k])]
        prop_1_0_1 = material_property[wp.int32(solid_id[0, i, j - 1, k])]
        prop_1_1_0 = material_property[wp.int32(solid_id[0, i, j, k - 1])]

        # Get average property
        if prop_1_1_1 + prop_1_1_0 == 0.0:
            prop_x = 0.0
        else:
            prop_x = (2.0 * prop_1_1_1 * prop_0_1_1) / (prop_1_1_1 + prop_0_1_1)
        if prop_1_1_1 + prop_1_0_1 == 0.0:
            prop_y = 0.0
        else:
            prop_y = (2.0 * prop_1_1_1 * prop_1_0_1) / (prop_1_1_1 + prop_1_0_1)
        if prop_1_1_1 + prop_0_1_1 == 0.0:
            prop_z = 0.0
        else:
            prop_z = (2.0 * prop_1_1_1 * prop_1_1_0) / (prop_1_1_1 + prop_1_1_0)

        return wp.vec3(prop_x, prop_y, prop_z)

    @wp.kernel
    def _update_magnetic_field(
        magnetic_field: wp.array4d(dtype=wp.float32),
        electric_field: wp.array4d(dtype=wp.float32),
        solid_id: wp.array4d(dtype=wp.uint8),
        mu_mapping: wp.array(dtype=wp.float32),
        sigma_m_mapping: wp.array(dtype=wp.float32),
        spacing: wp.vec3f,
        dt: wp.float32,
        nr_ghost_cells: wp.int32,
    ):
        # get index
        i, j, k = wp.tid()

        # Skip ghost cells
        i += +nr_ghost_cells
        j += +nr_ghost_cells
        k += +nr_ghost_cells

        # get properties
        mu = MagneticFieldUpdate._sample_magnetic_property(
            solid_id, mu_mapping, i, j, k
        )
        sigma_m = MagneticFieldUpdate._sample_magnetic_property(
            solid_id, sigma_m_mapping, i, j, k
        )

        # Get coefficients
        _denom = 2.0 * mu + sigma_m * dt
        c_hh = wp.cw_div(2.0 * mu - sigma_m * dt, _denom)
        c_he = (2.0 * dt) / wp.cw_mul(spacing, _denom)

        # Get electric field stencil
        e_x_0_0_0 = electric_field[0, i, j, k]
        e_x_0_1_0 = electric_field[0, i, j + 1, k]
        e_x_0_0_1 = electric_field[0, i, j, k + 1]
        e_y_0_0_0 = electric_field[1, i, j, k]
        e_y_1_0_0 = electric_field[1, i + 1, j, k]
        e_y_0_0_1 = electric_field[1, i, j, k + 1]
        e_z_0_0_0 = electric_field[2, i, j, k]
        e_z_1_0_0 = electric_field[2, i + 1, j, k]
        e_z_0_1_0 = electric_field[2, i, j + 1, k]

        # Apply zero flux boundary conditions
        #if i == electric_field.shape[1] - 2:
        #    e_y_1_0_0 = e_y_0_0_0
        #    e_z_1_0_0 = e_z_0_0_0
        #if j == electric_field.shape[2] - 2:
        #    e_x_0_1_0 = e_x_0_0_0
        #    e_z_0_1_0 = e_z_0_0_0
        #if k == electric_field.shape[3] - 2:
        #    e_x_0_0_1 = e_x_0_0_0
        #    e_y_0_0_1 = e_y_0_0_0



        # Get curl of electric field
        curl_e_x = (e_y_0_0_1 - e_y_0_0_0) - (e_z_0_1_0 - e_z_0_0_0)
        curl_e_y = (e_z_1_0_0 - e_z_0_0_0) - (e_x_0_0_1 - e_x_0_0_0)
        curl_e_z = (e_x_0_1_0 - e_x_0_0_0) - (e_y_1_0_0 - e_y_0_0_0)
        curl_e = wp.vec3(curl_e_x, curl_e_y, curl_e_z)

        # compute new magnetic field
        h = wp.vec3f(
            magnetic_field[0, i, j, k],
            magnetic_field[1, i, j, k],
            magnetic_field[2, i, j, k],
        )
        new_h = wp.cw_mul(c_hh, h) + wp.cw_mul(c_he, curl_e)

        # Set magnetic field
        magnetic_field[0, i, j, k] = new_h[0]
        magnetic_field[1, i, j, k] = new_h[1]
        magnetic_field[2, i, j, k] = new_h[2]

    def __call__(
        self,
        magnetic_field: wp.array4d(dtype=wp.float32),
        electric_field: wp.array4d(dtype=wp.float32),
        solid_id: wp.array4d(dtype=wp.uint8),
        mu_mapping: wp.array(dtype=wp.float32),
        sigma_m_mapping: wp.array(dtype=wp.float32),
        spacing: Union[float, tuple[float, float, float]],
        dt: float,
        nr_ghost_cells: int = 1,
    ):
        # Launch kernel
        wp.launch(
            self._update_magnetic_field,
            inputs=[
                magnetic_field,
                electric_field,
                solid_id,
                mu_mapping,
                sigma_m_mapping,
                spacing,
                dt,
                nr_ghost_cells,
            ],
            dim=[x - 2 * nr_ghost_cells for x in solid_id.shape[1:]],
        )

        return magnetic_field

class SetElectricField(Operator):
    """
    Sets the electric field to a given value
    """

    @wp.kernel
    def _set_electric_field(
        electric_field: wp.array4d(dtype=wp.float32),
        solid_id: wp.array4d(dtype=wp.uint8),
        id_number: wp.uint8,
        e: wp.float32,
        dim: wp.float32,
        nr_ghost_cells: wp.int32,
    ):
        # get index
        i, j, k = wp.tid()

        # Skip ghost cells
        i += nr_ghost_cells
        j += nr_ghost_cells
        k += nr_ghost_cells

        # Check if material id matches
        if solid_id[0, i, j, k] == id_number:
            if dim == 0:
                electric_field[0, i, j, k] = e
                electric_field[0, i, j+1, k] = e
                electric_field[0, i, j, k+1] = e
                electric_field[0, i, j+1, k+1] = e
            elif dim == 1:
                electric_field[1, i, j, k] = e
                electric_field[1, i+1, j, k] = e
                electric_field[1, i, j, k+1] = e
                electric_field[1, i+1, j, k+1] = e
            elif dim == 2:
                electric_field[2, i, j, k] = e
                electric_field[2, i+1, j, k] = e
                electric_field[2, i, j+1, k] = e
                electric_field[2, i+1, j+1, k] = e

    def __call__(
        self,
        electric_field: wp.array4d(dtype=wp.float32),
        solid_id: wp.array4d(dtype=wp.uint8),
        id_number: int,
        e: float,
        dim: int,
        nr_ghost_cells: int = 1,
    ):
        # Launch kernel
        wp.launch(
            self._set_electric_field,
            inputs=[electric_field, solid_id, id_number, e, dim, nr_ghost_cells],
            dim=[x - 2 * nr_ghost_cells for x in solid_id.shape[1:]],
        )

        return electric_field

class ElectricFieldToChargeDensity(Operator):
   """
   Electric field to charge density operator

   Basically a gaussian surface around the charge density cell
   """

   @wp.kernel
   def _electric_field_to_charge_density(
       electric_field: wp.array4d(dtype=wp.float32),
       charge_density: wp.array4d(dtype=wp.float32),
       solid_id: wp.array4d(dtype=wp.uint8),
       eps_mapping: wp.array(dtype=wp.float32),
       spacing: wp.vec3f,
       nr_ghost_cells: wp.int32,
   ):
       # get index
       i, j, k = wp.tid()

       # Skip ghost cells
       i += + nr_ghost_cells
       j += + nr_ghost_cells
       k += + nr_ghost_cells

       # get eps for each direction
       eps = ElectricFieldUpdate._sample_electric_property(solid_id, eps_mapping, i, j, k)
       eps_x_u = eps[0]
       eps_x_d = ElectricFieldUpdate._sample_electric_property(solid_id, eps_mapping, i - 1, j, k)[0]
       eps_y_u = eps[1]
       eps_y_d = ElectricFieldUpdate._sample_electric_property(solid_id, eps_mapping, i, j - 1, k)[1]
       eps_z_u = eps[2]
       eps_z_d = ElectricFieldUpdate._sample_electric_property(solid_id, eps_mapping, i, j, k - 1)[2]

       # Get electric field for each direction
       e_x_u = electric_field[0, i, j, k]
       e_x_d = electric_field[0, i-1, j, k]
       e_y_u = electric_field[1, i, j, k]
       e_y_d = electric_field[1, i, j-1, k]
       e_z_u = electric_field[2, i, j, k]
       e_z_d = electric_field[2, i, j, k-1]

       # Sum electric field dot normal
       charge_density[0, i, j, k] = (
           eps_x_u * e_x_u * spacing[1] * spacing[2]
           - eps_x_d * e_x_d * spacing[1] * spacing[2]
           + eps_y_u * e_y_u * spacing[0] * spacing[2]
           - eps_y_d * e_y_d * spacing[0] * spacing[2]
           + eps_z_u * e_z_u * spacing[0] * spacing[1]
           - eps_z_d * e_z_d * spacing[0] * spacing[1]
       ) / (spacing[0] * spacing[1] * spacing[2])

   def __call__(
       self,
       electric_field: wp.array4d(dtype=wp.float32),
       charge_density: wp.array4d(dtype=wp.float32),
       solid_id: wp.array4d(dtype=wp.uint8),
       eps_mapping: wp.array(dtype=wp.float32),
       spacing: Union[float, tuple[float, float, float]],
       nr_ghost_cells: int = 1,
   ):
       # Launch kernel
       wp.launch(
           self._electric_field_to_charge_density,
           inputs=[
               electric_field,
               charge_density,
               solid_id,
               eps_mapping,
               spacing,
               nr_ghost_cells,
           ],
           dim=[x - 2 * nr_ghost_cells for x in solid_id.shape[1:]],
       )

       return charge_density

class ComputeElectricEnergy(Operator):
    """
    Compute electric energy
    """

    @wp.kernel
    def _total_electric_energy(
        total_energy: wp.array(dtype=wp.float32),
        electric_field: wp.array4d(dtype=wp.float32),
        solid_id: wp.array4d(dtype=wp.uint8),
        eps_mapping: wp.array(dtype=wp.float32),
        spacing: wp.vec3f,
        nr_ghost_cells: wp.int32,
    ):
        # get index
        i, j, k = wp.tid()

        # Skip ghost cells
        i += nr_ghost_cells
        j += nr_ghost_cells
        k += nr_ghost_cells

        # get eps for each direction
        eps = ElectricFieldUpdate._sample_electric_property(solid_id, eps_mapping, i, j, k)

        # Get electric field for each direction
        e_x = electric_field[0, i, j, k]
        e_y = electric_field[1, i, j, k]
        e_z = electric_field[2, i, j, k]

        # Compute energy
        cell_volume = spacing[0] * spacing[1] * spacing[2]
        energy = 0.5 * cell_volume * (
            eps[0] * e_x * e_x
            + eps[1] * e_y * e_y
            + eps[2] * e_z * e_z
        )

        # Atomic add to total energy
        wp.atomic_add(total_energy, 0, energy)

    def __call__(
        self,
        electric_field: wp.array4d(dtype=wp.float32),
        solid_id: wp.array4d(dtype=wp.uint8),
        eps_mapping: wp.array(dtype=wp.float32),
        spacing: Union[float, tuple[float, float, float]],
        nr_ghost_cells: int = 1,
    ):
        # Allocate total energy
        total_energy = wp.zeros(1, dtype=wp.float32)

        # Launch kernel
        wp.launch(
            self._total_electric_energy,
            inputs=[
                total_energy,
                electric_field,
                solid_id,
                eps_mapping,
                spacing,
                nr_ghost_cells,
            ],
            dim=[x - 2 * nr_ghost_cells for x in solid_id.shape[1:]],
        )

        return total_energy.numpy()[0]

class ComputeMagneticEnergy(Operator):
    """
    Compute magnetic energy
    """

    @wp.kernel
    def _total_magnetic_energy(
        total_energy: wp.array(dtype=wp.float32),
        magnetic_field: wp.array4d(dtype=wp.float32),
        solid_id: wp.array4d(dtype=wp.uint8),
        mu_mapping: wp.array(dtype=wp.float32),
        spacing: wp.vec3f,
        nr_ghost_cells: wp.int32,
    ):
        # get index
        i, j, k = wp.tid()

        # Skip ghost cells
        i += nr_ghost_cells
        j += nr_ghost_cells
        k += nr_ghost_cells

        # get mu for each direction
        mu = MagneticFieldUpdate._sample_magnetic_property(solid_id, mu_mapping, i, j, k)

        # Get magnetic field for each direction
        h_x = magnetic_field[0, i, j, k]
        h_y = magnetic_field[1, i, j, k]
        h_z = magnetic_field[2, i, j, k]

        # Compute energy
        cell_volume = spacing[0] * spacing[1] * spacing[2]
        energy = 0.5 * cell_volume * (
            mu[0] * h_x * h_x
            + mu[1] * h_y * h_y
            + mu[2] * h_z * h_z
        )

        # Atomic add to total energy
        wp.atomic_add(total_energy, 0, energy)

    def __call__(
        self,
        magnetic_field: wp.array4d(dtype=wp.float32),
        solid_id: wp.array4d(dtype=wp.uint8),
        mu_mapping: wp.array(dtype=wp.float32),
        spacing: Union[float, tuple[float, float, float]],
        nr_ghost_cells: int = 1,
    ):
        # Allocate total energy
        total_energy = wp.zeros(1, dtype=wp.float32)

        # Launch kernel
        wp.launch(
            self._total_magnetic_energy,
            inputs=[
                total_energy,
                magnetic_field,
                solid_id,
                mu_mapping,
                spacing,
                nr_ghost_cells,
            ],
            dim=[x - 2 * nr_ghost_cells for x in solid_id.shape[1:]],
        )

        return total_energy.numpy()[0]


class SumChargeDensity(Operator):
   """
   Sums the charge density for each of the solid ids
   """

   @wp.kernel
   def _electric_field_to_charge_density(
       charge_density: wp.array4d(dtype=wp.float32),
       solid_id: wp.array4d(dtype=wp.uint8),
       total_surface_charge: wp.array(dtype=wp.float32),
       total_volume_charge: wp.array(dtype=wp.float32),
       spacing: wp.vec3f,
       nr_ghost_cells: wp.int32,
   ):
       # get index
       i, j, k = wp.tid()

       # Skip ghost cells
       i += + nr_ghost_cells
       j += + nr_ghost_cells
       k += + nr_ghost_cells

       # Get charge density and solid id
       charge = charge_density[0, i, j, k] * spacing[0] * spacing[1] * spacing[2]
       id_1_1_1 = wp.int32(solid_id[0, i, j, k])
       id_0_1_1 = wp.int32(solid_id[0, i - 1, j, k])
       id_1_0_1 = wp.int32(solid_id[0, i, j - 1, k])
       id_0_0_1 = wp.int32(solid_id[0, i - 1, j - 1, k])
       id_1_1_0 = wp.int32(solid_id[0, i, j, k - 1])
       id_0_1_0 = wp.int32(solid_id[0, i - 1, j, k - 1])
       id_1_0_0 = wp.int32(solid_id[0, i, j - 1, k - 1])
       id_0_0_0 = wp.int32(solid_id[0, i - 1, j - 1, k - 1])

       # Check if volume charge
       if (id_1_1_1 == id_0_1_1 == id_1_0_1 == id_0_0_1 == id_1_1_0 == id_0_1_0 == id_1_0_0 == id_0_0_0):
           wp.atomic_add(total_volume_charge, id_1_1_1, charge)

       # Add to any of the surface charges
       for id in range(total_surface_charge.shape[0]):
           if id_1_1_1 == id or id_0_1_1 == id or id_1_0_1 == id or id_0_0_1 == id or id_1_1_0 == id or id_0_1_0 == id or id_1_0_0 == id or id_0_0_0 == id:
               wp.atomic_add(total_surface_charge, id, charge)


   def __call__(
         self,
         charge_density: wp.array4d(dtype=wp.float32),
         solid_id: wp.array4d(dtype=wp.uint8),
         total_surface_charge: wp.array(dtype=wp.float32),
         total_volume_charge: wp.array(dtype=wp.float32),
         spacing: Union[float, tuple[float, float, float]],
         nr_ghost_cells: int = 1,
    ):
         # Launch kernel
         wp.launch(
              self._electric_field_to_charge_density,
              inputs=[
                charge_density,
                solid_id,
                total_surface_charge,
                total_volume_charge,
                spacing,
                nr_ghost_cells,
              ],
              dim=[x - 2 * nr_ghost_cells for x in solid_id.shape[1:]],
         )
    
         return total_surface_charge, total_volume_charge
