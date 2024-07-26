# FDTD solver for 3D Yee grid

from dataclasses import dataclass

import warp as wp
import numpy as np

from dense_plasma_focus.material import get_materials_in_compound, Material

wp.init()


def em_cfl(dx: float, c: float = 299792458.0) -> float:
    return dx / (c * np.sqrt(3.0))


@wp.func
def sample_electric_property(
    material_id: wp.array3d(dtype=wp.uint8),
    material_property: wp.array(dtype=wp.float32),
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
):
    # Get material property
    prop_0_0_1 = material_property[wp.int32(material_id[i - 1, j - 1, k])]
    prop_0_1_0 = material_property[wp.int32(material_id[i - 1, j, k - 1])]
    prop_0_1_1 = material_property[wp.int32(material_id[i - 1, j, k])]
    prop_1_0_0 = material_property[wp.int32(material_id[i, j - 1, k - 1])]
    prop_1_0_1 = material_property[wp.int32(material_id[i, j - 1, k])]
    prop_1_1_0 = material_property[wp.int32(material_id[i, j, k - 1])]
    prop_1_1_1 = material_property[wp.int32(material_id[i, j, k])]

    # Get average property
    prop_x = (prop_1_1_1 + prop_1_1_0 + prop_1_0_1 + prop_1_0_0) / 4.0
    prop_y = (prop_1_1_1 + prop_1_1_0 + prop_0_1_1 + prop_0_1_0) / 4.0
    prop_z = (prop_1_1_1 + prop_1_0_1 + prop_0_1_1 + prop_0_0_1) / 4.0

    return wp.vec3(prop_x, prop_y, prop_z)


@wp.func
def sample_magnetic_property(
    material_id: wp.array3d(dtype=wp.uint8),
    material_property: wp.array(dtype=wp.float32),
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
):
    # Get material property
    prop_1_1_1 = material_property[wp.int32(material_id[i, j, k])]
    prop_0_1_1 = material_property[wp.int32(material_id[i - 1, j, k])]
    prop_1_0_1 = material_property[wp.int32(material_id[i, j - 1, k])]
    prop_1_1_0 = material_property[wp.int32(material_id[i, j, k - 1])]

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


def construct_em_update_kernels():
    """
    Construct kernels for updating electric and magnetic fields
    """

    # Make kernel for updating electric field
    @wp.kernel
    def _update_eletric_field(
        electric_field_x: wp.array3d(dtype=wp.float32),
        electric_field_y: wp.array3d(dtype=wp.float32),
        electric_field_z: wp.array3d(dtype=wp.float32),
        magnetic_field_x: wp.array3d(dtype=wp.float32),
        magnetic_field_y: wp.array3d(dtype=wp.float32),
        magnetic_field_z: wp.array3d(dtype=wp.float32),
        impressed_current_x: wp.array3d(dtype=wp.float32),
        impressed_current_y: wp.array3d(dtype=wp.float32),
        impressed_current_z: wp.array3d(dtype=wp.float32),
        material_id: wp.array3d(dtype=wp.uint8),
        eps_mapping: wp.array(dtype=wp.float32),
        sigma_e_mapping: wp.array(dtype=wp.float32),
        dx: wp.float32,
        dt: wp.float32,
    ):
        # get index
        i, j, k = wp.tid()

        # Skip ghost cells
        i += 1
        j += 1
        k += 1

        # get properties
        eps = sample_electric_property(material_id, eps_mapping, i, j, k)
        sigma_e = sample_electric_property(material_id, sigma_e_mapping, i, j, k)

        # Get coefficients
        _denom = 2.0 * eps + sigma_e * dt
        c_ee = wp.cw_div(2.0 * eps - sigma_e * dt, _denom)
        c_eh = (2.0 * dt) / (dx * _denom)
        c_ej = (-2.0 * dt) / _denom

        # Get curl of magnetic field
        curl_h_x = (magnetic_field_z[i, j, k] - magnetic_field_z[i, j - 1, k]) - (
            magnetic_field_y[i, j, k] - magnetic_field_y[i, j, k - 1]
        )
        curl_h_y = (magnetic_field_x[i, j, k] - magnetic_field_x[i, j, k - 1]) - (
            magnetic_field_z[i, j, k] - magnetic_field_z[i - 1, j, k]
        )
        curl_h_z = (magnetic_field_y[i, j, k] - magnetic_field_y[i - 1, j, k]) - (
            magnetic_field_x[i, j, k] - magnetic_field_x[i, j - 1, k]
        )
        curl_h = wp.vec3(curl_h_x, curl_h_y, curl_h_z)

        # compute new electric field
        e = wp.vec3f(
            electric_field_x[i, j, k],
            electric_field_y[i, j, k],
            electric_field_z[i, j, k],
        )
        cur = wp.vec3f(
            impressed_current_x[i, j, k],
            impressed_current_y[i, j, k],
            impressed_current_z[i, j, k],
        )
        new_e = wp.cw_mul(c_ee, e) + wp.cw_mul(c_eh, curl_h) + wp.cw_mul(c_ej, cur)

        # Set electric field
        electric_field_x[i, j, k] = new_e[0]
        electric_field_y[i, j, k] = new_e[1]
        electric_field_z[i, j, k] = new_e[2]

    @wp.kernel
    def _update_magnetic_field(
        electric_field_x: wp.array3d(dtype=wp.float32),
        electric_field_y: wp.array3d(dtype=wp.float32),
        electric_field_z: wp.array3d(dtype=wp.float32),
        magnetic_field_x: wp.array3d(dtype=wp.float32),
        magnetic_field_y: wp.array3d(dtype=wp.float32),
        magnetic_field_z: wp.array3d(dtype=wp.float32),
        material_id: wp.array3d(dtype=wp.uint8),
        mu_mapping: wp.array(dtype=wp.float32),
        sigma_m_mapping: wp.array(dtype=wp.float32),
        dx: wp.float32,
        dt: wp.float32,
    ):
        # get index
        i, j, k = wp.tid()

        # Skip ghost cells
        i += 1
        j += 1
        k += 1

        # get properties
        mu = sample_electric_property(material_id, mu_mapping, i, j, k)
        sigma_m = sample_electric_property(material_id, sigma_m_mapping, i, j, k)

        # Get coefficients
        _denom = 2.0 * mu + sigma_m * dt
        c_hh = wp.cw_div(2.0 * mu - sigma_m * dt, _denom)
        c_he = (2.0 * dt) / (dx * _denom)

        # Get curl of electric field
        curl_e_x = (electric_field_y[i, j, k + 1] - electric_field_y[i, j, k]) - (
            electric_field_z[i, j + 1, k] - electric_field_z[i, j, k]
        )
        curl_e_y = (electric_field_z[i + 1, j, k] - electric_field_z[i, j, k]) - (
            electric_field_x[i, j, k + 1] - electric_field_x[i, j, k]
        )
        curl_e_z = (electric_field_x[i, j + 1, k] - electric_field_x[i, j, k]) - (
            electric_field_y[i + 1, j, k] - electric_field_y[i, j, k]
        )
        curl_e = wp.vec3(curl_e_x, curl_e_y, curl_e_z)

        # compute new magnetic field
        h = wp.vec3f(
            magnetic_field_x[i, j, k],
            magnetic_field_y[i, j, k],
            magnetic_field_z[i, j, k],
        )
        new_h = wp.cw_mul(c_hh, h) + wp.cw_mul(c_he, curl_e)

        # Set magnetic field
        magnetic_field_x[i, j, k] = new_h[0]
        magnetic_field_y[i, j, k] = new_h[1]
        magnetic_field_z[i, j, k] = new_h[2]

    return _update_eletric_field, _update_magnetic_field


if __name__ == "__main__":
    # imports
    from dense_plasma_focus.reactor.reactor import LLPReactor
    from dense_plasma_focus.solver.voxelizer import voxelize_compound

    # Create reactor
    reactor = LLPReactor()
    # reactor.export_step("reactor.step")
    reactor.export_3mf("reactor.3mf")

    # Get bounding box
    bounding_box = reactor.bounding_box()
    origin = (bounding_box.min.X, bounding_box.min.Y, bounding_box.min.Z)
    dx = 0.5
    internal_voxels = (
        int((bounding_box.max.X - bounding_box.min.X) / dx),
        int((bounding_box.max.Y - bounding_box.min.Y) / dx),
        int((bounding_box.max.Z - bounding_box.min.Z) / dx),
    )
    dt = em_cfl(dx)

    # Get number of voxels including ghost cells
    nr_voxels = (
        internal_voxels[0] + 2,
        internal_voxels[1] + 2,
        internal_voxels[2] + 2,
    )

    # Voxelize electrode
    materials = get_materials_in_compound(reactor)
    material_id = voxelize_compound(
        compound=reactor,
        origin=(
            origin[0] - dx / 2.0,
            origin[1] - dx / 2.0,
            origin[2] - dx / 2.0,
        ),  # shift origin by half a voxel to center on ghost cell
        resolution=dx,
        nr_voxels=nr_voxels,
        materials=materials,
    )

    # Make electric field
    electric_field_x = wp.zeros(nr_voxels, wp.float32)
    electric_field_y = wp.zeros(nr_voxels, wp.float32)
    electric_field_z = wp.zeros(nr_voxels, wp.float32)

    # Make magnetic field
    magnetic_field_x = wp.zeros(nr_voxels, wp.float32)
    magnetic_field_y = wp.zeros(nr_voxels, wp.float32)
    magnetic_field_z = wp.zeros(nr_voxels, wp.float32)

    # Make impressed current
    impressed_electric_current_x = wp.zeros(nr_voxels, wp.float32)
    impressed_electric_current_y = wp.zeros(nr_voxels, wp.float32)
    impressed_electric_current_z = wp.zeros(nr_voxels, wp.float32)

    # Make material properties
    eps_mapping = wp.from_numpy(
        np.array([m.eps for m in materials], dtype=np.float32),
        dtype=wp.float32,
    )
    sigma_e_mapping = wp.from_numpy(
        np.array([m.sigma_e for m in materials], dtype=np.float32),
        dtype=wp.float32,
    )

    # Get em kernels
    (
        update_electric_field_kernel,
        update_magnetic_field_kernel,
    ) = construct_em_update_kernels()

    # Run kernel
    from tqdm import tqdm
    import time

    nr_steps = 1000
    tic = time.time()
    for _ in tqdm(range(nr_steps)):
        wp.launch(
            update_electric_field_kernel,
            inputs=[
                electric_field_x,
                electric_field_y,
                electric_field_z,
                magnetic_field_x,
                magnetic_field_y,
                magnetic_field_z,
                impressed_electric_current_x,
                impressed_electric_current_x,
                impressed_electric_current_x,
                material_id,
                eps_mapping,
                sigma_e_mapping,
                dx,
                dt,
            ],
            dim=internal_voxels,
        )
        wp.launch(
            update_magnetic_field_kernel,
            inputs=[
                electric_field_x,
                electric_field_y,
                electric_field_z,
                magnetic_field_x,
                magnetic_field_y,
                magnetic_field_z,
                material_id,
                eps_mapping,
                sigma_e_mapping,
                dx,
                dt,
            ],
            dim=internal_voxels,
        )
    wp.synchronize()
    toc = time.time()
    print("number of voxels:", internal_voxels)
    print("time:", toc - tic)
    print(
        "million voxels per second:",
        nr_steps * np.prod(internal_voxels) / (toc - tic) / 1e6,
    )
