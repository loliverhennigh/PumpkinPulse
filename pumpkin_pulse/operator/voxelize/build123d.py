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

from pumpkin_pulse.compute_backend import ComputeBackend
from pumpkin_pulse.material import get_materials_in_compound, Material, VACUUM
from pumpkin_pulse.operator.voxelize.voxelize import Voxelize


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


class Build123DVoxelize(Voxelize):
    """
    Voxelize a build123d compound
    """

    def __call__(
        self,
        voxel_id: wp.array4d(dtype=wp.uint8),
        geometry: Compound,
        id_number: int,
        spacing: Union[float, tuple[float, float, float]],
        origin: tuple[float, float, float],
        nr_ghost_cells: int = 1,
        tolerance: float = 0.001,
        angular_tolerance: float = 0.1,
        nr_processes: int = 16,
    ):
        # Export geometry to stl
        with tempfile.NamedTemporaryFile(suffix=".stl") as f:
            geometry.export_stl(
                f.name, tolerance=tolerance, angular_tolerance=angular_tolerance
            )
            mesh = np_mesh.Mesh.from_file(f.name)
            mesh_points = mesh.points.reshape(-1, 3)
            mesh_indices = np.arange(mesh_points.shape[0])
            mesh = wp.Mesh(
                points=wp.array(mesh_points, dtype=wp.vec3),
                indices=wp.array(mesh_indices, dtype=int),
            )

        # Voxelize STL of mesh
        wp.launch(
            self._voxelize_mesh,
            inputs=[
                voxel_id,
                mesh.id,
                id_number,
                spacing,
                origin,
                nr_ghost_cells,
            ],
            dim=[d - 2 * nr_ghost_cells for d in voxel_id.shape[1:]],
        )

        # Process points near the boundary
        np_voxel_id = voxel_id.numpy()

        # Export geometry to brep
        with tempfile.NamedTemporaryFile(suffix=".brep") as f:
            # Get brep
            geometry.export_brep(f.name)

            # Get all points with 255 material id and indexes
            indices = np.argwhere(np_voxel_id[0] == 255)
            points = (indices + 0.5) * np.array(spacing) + np.array(origin)

            # Check if points are inside or outside
            if len(points) != 0:

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
                subset = indices[results]
                np_voxel_id[0, subset[:, 0], subset[:, 1], subset[:, 2]] = id_number

        # Set all 255 to 0
        np_voxel_id[np_voxel_id == 255] = 0

        # Set voxel id
        voxel_id = wp.from_numpy(np_voxel_id, dtype=wp.uint8)

        return voxel_id
