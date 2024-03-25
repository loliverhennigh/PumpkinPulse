# Voxelize a build123d compound

from typing import Literal, Union
import tempfile
from stl import mesh as np_mesh
import numpy as np
from anytree.search import findall
from build123d import Part, Compound
import cupy as cp
from multiprocessing import Pool
from math import ceil
from tqdm import tqdm

from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.TopoDS import TopoDS_Shape
from OCP.gp import gp_Pnt
from OCP.BRep import BRep_Builder
from OCP.BRepTools import BRepTools
from OCP.TopAbs import TopAbs_OUT

import warp as wp

from dense_plasma_focus.compute_backend import ComputeBackend
from dense_plasma_focus.material import get_materials_in_compound, Material, VACUUM
from dense_plasma_focus.operator.voxelize.voxelize import Voxelize

# Create batch processor
def _process_batch(filename, batch):
    brep_shape = TopoDS_Shape()
    builder = BRep_Builder()
    BRepTools.Read_s(brep_shape, filename, builder)
    results = np.zeros(batch.shape[0], dtype=np.bool_)
    for i, point in enumerate(batch):
        classifier = BRepClass3d_SolidClassifier()
        classifier.Load(brep_shape)
        classifier.Perform(gp_Pnt(*point), 1e-6)
        state = classifier.State()
        results[i] = state != TopAbs_OUT
    return results

def get_materials_in_compound(
    compound: Compound
):
    """
    Get all materials in a build123d compound

    Parameters
    ----------
    compound : Compound
        Build123d compound
    """

    materials = []
    for part in findall(compound, filter_=lambda node: isinstance(node, Part)):
        materials.append(part.material)
    materials = list(set(materials))
    materials.sort(key=lambda x: x.name)

    # Make sure vacuum is always first
    if VACUUM in materials:
        materials.remove(VACUUM)
    materials.insert(0, VACUUM)

    return materials


class Build123DVoxelize(Voxelize):
    """
    Voxelize a build123d compound
    """

    def __call__(
        self,
        voxels: wp.array3d(dtype=wp.uint8),
        compound: Compound,
        spacing: Union[float, tuple[float, float, float]],
        origin: tuple[float, float, float],
        shape: tuple[int, int, int],
        materials: list[Material],
        nr_ghost_cells: int = 1,
        tolerance: float = 0.001,
        angular_tolerance: float = 0.1,
        nr_processes: int = 16,
    ):

        # Voxelize all parts
        parts = list(findall(compound, filter_=lambda node: isinstance(node, Part)))
        for i, part in tqdm(list(enumerate(parts)), desc="Voxelizing parts"):

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

            # Voxelize STL of mesh
            wp.launch(
                self._voxelize_mesh,
                inputs=[
                    voxels,
                    mesh.id,
                    spacing,
                    origin,
                    shape,
                    max([s * ss for s, ss in zip(shape, spacing)]),
                    material_id,
                ],
                dim=shape,
            )

            # Process points near the boundary
            np_voxels = voxels.numpy()

            # Export part to brep
            with tempfile.NamedTemporaryFile(suffix=".brep") as f:
                # Get brep
                part.export_brep(f.name)

                # Get all points with 255 material id and indexes
                indices = np.argwhere(np_voxels == 255)
                points = (indices + 0.5) * np.array(spacing) + np.array(origin)

                if len(points) == 0:
                    continue

                # Process points in batches
                batch_size = ceil(len(points) / nr_processes)
                with Pool(nr_processes) as p:
                    results = p.starmap(
                        _process_batch,
                        [
                            (f.name, points[i : i + batch_size])
                            for i in range(0, len(points), batch_size)
                        ],
                    )
                results = np.concatenate(results)

                # Set points to material id if inside
                np_voxels[tuple(indices[results].T)] = material_id

            # If last part, set all 255 to vacuum (0)
            if i == len(parts) - 1:
                np_voxels[np_voxels == 255] = 0

            # Set voxels
            voxels = wp.from_numpy(np_voxels, dtype=wp.uint8)

        return voxels
