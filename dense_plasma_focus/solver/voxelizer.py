# Voxelize a build123d compound

from typing import Literal, Union
import tempfile
from stl import mesh as np_mesh
import numpy as np
from anytree.search import findall
from build123d import Part, Compound
import cupy as cp

import warp as wp

from dense_plasma_focus.material import get_materials_in_compound, Material

wp.init()

@wp.kernel
def _voxelize_mesh(
    voxels: wp.array3d(dtype=wp.uint8),
    mesh: wp.uint64,
    dx: float,
    start_x: float,
    start_y: float,
    start_z: float,
    max_length: float,
    material_id: int
):

    # get index of voxel
    i, j, k = wp.tid()

    # position of voxel
    r_x = wp.float(i) * dx + start_x
    r_y = wp.float(j) * dx + start_y
    r_z = wp.float(k) * dx + start_z
    pos = wp.vec3(r_x, r_y, r_z)

    # evaluate distance of point
    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)
    if (wp.mesh_query_point(mesh, pos, max_length, sign, face_index, face_u, face_v)):
        p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
        delta = pos-p
        norm = wp.sqrt(wp.dot(delta, delta))

        # set point to be solid
        if (sign < 0) or (norm < dx/4.0): # TODO: fix this
            voxels[i, j, k] = wp.uint8(material_id)

def voxelize_compound(
    compound: Compound,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    resolution: float = 0.1,
    nr_voxels: tuple[int, int, int] = None,
    voxels: wp.array3d(dtype=wp.uint8) = None,
    materials: list[Material] = None,
    tolerance: float = 0.001,
    angular_tolerance: float = 0.1,
):
    """Voxelize a build123d compound

    Parameters
    ----------
    compound : Compound
        Build123d compound
    origin : tuple[float, float, float], optional
        Origin of the voxel grid, by default (0.0, 0.0, 0.0)
    resolution : float, optional
        Resolution of the voxel grid, by default 0.1
    nr_voxels : tuple[int, int, int], optional
        Number of voxels in each direction, by default None
    voxels : wp.array3d(dtype=wp.uint8), optional
        Voxel grid, by default None
    materials : list[Material], optional
        List of materials in the compound, by default None
    tolerance : float, optional
        Tolerance for stl export, by default 0.001
    angular_tolerance : float, optional
        Angular tolerance for stl export, by default 0.1

    Returns
    -------
    wp.array3d(dtype=wp.uint8)
        Voxel grid
    """

    # Get voxels if not provided
    if voxels is None:
        voxels = wp.zeros(
            shape=nr_voxels,
            dtype=wp.uint8,
        )

    # Voxelize all parts
    for part in findall(compound, filter_=lambda node: isinstance(node, Part)):

        # Move part to correct position
        part_parent = part.parent
        while part_parent is not None:
            part = part_parent.location * part
            part_parent = part_parent.parent

        # Export part to stl
        with tempfile.NamedTemporaryFile(suffix=".stl") as f:
            part.export_stl(f.name, tolerance=tolerance, angular_tolerance=angular_tolerance)
            mesh = np_mesh.Mesh.from_file(f.name)
            mesh_points = mesh.points.reshape(-1, 3)
            mesh_indices = np.arange(mesh_points.shape[0])
            mesh = wp.Mesh(
                points=wp.array(mesh_points, dtype=wp.vec3),
                indices=wp.array(mesh_indices, dtype=int),
            )

        # Get material id
        material_id = materials.index(part.material)

        # Voxelize mesh
        wp.launch(
            _voxelize_mesh,
            inputs=[
                voxels,
                mesh.id,
                resolution,
                origin[0],
                origin[1],
                origin[2],
                max(nr_voxels) * resolution,
                material_id,
            ],
            dim=nr_voxels,
        )

    return voxels

if __name__ == "__main__":

    from dense_plasma_focus.reactor.reactor import LLPReactor
    from build123d import Line, CenterArc, Location, Compound
    import copy
    import phantomgaze as pg

    # Create reactor
    reactor = LLPReactor()

    # Get bounding box
    bounding_box = reactor.bounding_box()
    origin = (bounding_box.min.X, bounding_box.min.Y, bounding_box.min.Z)
    dx = 0.1
    nr_voxels = (
        int((bounding_box.max.X - bounding_box.min.X) / dx),
        int((bounding_box.max.Y - bounding_box.min.Y) / dx),
        int((bounding_box.max.Z - bounding_box.min.Z) / dx),
    )

    # Voxelize electrode
    materials = get_materials_in_compound(reactor)
    voxels = voxelize_compound(
        compound=reactor,
        origin=origin,
        resolution=dx,
        nr_voxels=nr_voxels,
        materials=materials,
    )

    # Render voxels
    voxel_volume = pg.objects.Volume(
        voxels,
        spacing=(dx, dx, dx),
        origin=origin,
    )

    # Camera
    camera = pg.Camera(
        position=(-20, -20, 40),
        focal_point=(0, 0, 10),
        view_up=(0, 1, 0),
        max_depth=100.0,
    )

    # Render
    colormap = pg.Colormap("jet", vmin=0.0, vmax=float(len(materials)), opacity=cp.linspace(0.0, 1.0, 256))
    screen_buffer = pg.render.contour(voxel_volume, camera, threshold=1.5, color=voxel_volume, colormap=colormap)
    screen_buffer = pg.render.contour(voxel_volume, camera, threshold=0.5, color=voxel_volume, colormap=colormap)

    # Show
    import matplotlib.pyplot as plt
    plt.imshow(screen_buffer.image.get())
    plt.show()

