# Simple example of voxelizing a build123d geometry

import warp as wp
import numpy as np
import matplotlib.pyplot as plt

wp.init()

from dense_plasma_focus.geometry.reactor.reactor import LLPReactor
from dense_plasma_focus.geometry.circuit.capacitor_discharge import CapacitorDischarge
from dense_plasma_focus.operator.voxelize.build123d import Build123D, get_materials_in_compound

if __name__ == "__main__":

    # Create geometry
    #reactor = LLPReactor()
    geometry = CapacitorDischarge()

    # Get bounding box
    bounding_box = geometry.bounding_box()
    dx = 0.25
    origin = (bounding_box.min.X, bounding_box.min.Y, bounding_box.min.Z)
    nr_voxels = (
        int(((bounding_box.max.X - bounding_box.min.X)) / dx),
        int(((bounding_box.max.Y - bounding_box.min.Y)) / dx),
        int(((bounding_box.max.Z - bounding_box.min.Z)) / dx),
    )

    # Get materials in geometry
    materials = get_materials_in_compound(geometry)

    # Make voxelizer
    voxelizer = Build123D()

    # Voxelize electrode
    voxels = wp.zeros(nr_voxels, wp.uint8)
    voxels = voxelizer(
        voxels,
        compound=geometry,
        spacing=(dx, dx, dx),
        origin=origin,
        shape=nr_voxels,
        materials=materials,
        nr_processes=32,
    )

    # Plot
    import matplotlib.pyplot as plt
    plt.imshow(voxels.numpy()[:, nr_voxels[1]//2, :])
    plt.show()
