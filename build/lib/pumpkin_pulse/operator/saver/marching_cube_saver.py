import os
import numpy as np
import pyvista as pv
import warp as wp

from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.operator.saver.field_saver import FieldSaver
from pumpkin_pulse.functional.marching_cube import VERTEX_TABLE, VERTEX_INDICES_TABLE

"""
class MarchingCubeSaver(Operator):

    # Make push kernel
    @wp.kernel
    def get_triangles(
        material_properties: MaterialProperties,
        triangles: wp.array2d(dtype=wp.float32),
        normals: wp.array2d(dtype=wp.float32),
        nr_triangles: wp.array(dtype=wp.int32),
        vertex_indices_table: wp.array2d(dtype=wp.int32),
        vertex_table: wp.array2d(dtype=wp.float32),
    ):

        # Get particle index
        i, j, k = wp.tid()

        # Store marching cubes id
        mc_id = material_properties.mc_id.data[i, j, k]

        # Check if any triangles in cell
        if vertex_indices_table[wp.int32(mc_id), 0] != -1:

            # Get origin of cell
            cell_origin = (
                material_properties.mc_id.origin
                + wp.cw_mul(
                    material_properties.mc_id.spacing,
                    (
                        wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k))
                    ),
                )
            )

            # For each cell check collisions with triangles (max 5 triangles)
            for _ti in range(5):

                # Get vertex index
                vertex_index_0 = vertex_indices_table[wp.int32(mc_id), _ti * 3 + 0]
                vertex_index_1 = vertex_indices_table[wp.int32(mc_id), _ti * 3 + 1]
                vertex_index_2 = vertex_indices_table[wp.int32(mc_id), _ti * 3 + 2]

                # Break if no triangle
                if vertex_index_0 == -1:
                    break
                else:

                    # Get triangle
                    vertex_0 = wp.vec3(
                        vertex_table[vertex_index_0, 0] * material_properties.mc_id.spacing[0] + cell_origin[0],
                        vertex_table[vertex_index_0, 1] * material_properties.mc_id.spacing[1] + cell_origin[1],
                        vertex_table[vertex_index_0, 2] * material_properties.mc_id.spacing[2] + cell_origin[2],
                    )
                    vertex_1 = wp.vec3(
                        vertex_table[vertex_index_1, 0] * material_properties.mc_id.spacing[0] + cell_origin[0],
                        vertex_table[vertex_index_1, 1] * material_properties.mc_id.spacing[1] + cell_origin[1],
                        vertex_table[vertex_index_1, 2] * material_properties.mc_id.spacing[2] + cell_origin[2],
                    )
                    vertex_2 = wp.vec3(
                        vertex_table[vertex_index_2, 0] * material_properties.mc_id.spacing[0] + cell_origin[0],
                        vertex_table[vertex_index_2, 1] * material_properties.mc_id.spacing[1] + cell_origin[1],
                        vertex_table[vertex_index_2, 2] * material_properties.mc_id.spacing[2] + cell_origin[2],
                    )

                    # Get normal
                    normal = wp.cross(vertex_1 - vertex_0, vertex_2 - vertex_0)
                    normal = wp.normalize(normal)

                    # Get triangle index
                    g = wp.atomic_add(nr_triangles, 0, 1)

                    # Add triangle to triangles
                    triangles[g * 3 + 0, 0] = vertex_0[0]
                    triangles[g * 3 + 0, 1] = vertex_0[1]
                    triangles[g * 3 + 0, 2] = vertex_0[2]
                    triangles[g * 3 + 1, 0] = vertex_1[0]
                    triangles[g * 3 + 1, 1] = vertex_1[1]
                    triangles[g * 3 + 1, 2] = vertex_1[2]
                    triangles[g * 3 + 2, 0] = vertex_2[0]
                    triangles[g * 3 + 2, 1] = vertex_2[1]
                    triangles[g * 3 + 2, 2] = vertex_2[2]

                    # Add normal to normals
                    normals[g, 0] = normal[0]
                    normals[g, 1] = normal[1]
                    normals[g, 2] = normal[2]

    def __call__(
        self,
        material_properties: MaterialProperties,
        filename: str,
        nr_triangles: int = None,
    ):

        # Get estimated number of triangles
        if nr_triangles is None:
            nr_triangles = 5 * np.prod(material_properties.mc_id.shape)

        # Allocate triangles
        triangles = wp.zeros((3 * nr_triangles, 3), dtype=wp.float32)
        normals = wp.zeros((nr_triangles, 3), dtype=wp.float32)
        nr_triangles = wp.zeros(1, dtype=wp.int32)

        # Launch kernel
        wp.launch(
            self.get_triangles,
            inputs=[
                material_properties,
                triangles,
                normals,
                nr_triangles,
                VERTEX_INDICES_TABLE,
                VERTEX_TABLE,
            ],
            dim=(material_properties.mc_id.shape[0], material_properties.mc_id.shape[1], material_properties.mc_id.shape[2]),
        )

        # Get numpy array
        np_triangles = triangles[:3*nr_triangles.numpy()[0]].numpy()
        np_normals = normals[:nr_triangles.numpy()[0]].numpy()
        del triangles
        del normals
        del nr_triangles

        # Get number of points
        nr_points = np_triangles.shape[0]

        # Create the cell array
        cells = np.hstack([[3, i, i + 1, i + 2] for i in range(0, nr_points, 3)])

        # Create the polydata
        mesh = pv.PolyData(np_triangles, cells)

        # Add normals
        mesh.cell_data["Normals"] = np_normals

        # Save the mesh
        mesh.save(filename)

        return None
"""
