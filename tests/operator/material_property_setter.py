import pytest
from build123d import Rectangle, extrude, Location

from pumpkin_pulse.operator.allocator import MaterialPropertyAllocator
from pumpkin_pulse.operator.material_property_setter import SphereMaterialPropertySetter, MeshMaterialPropertySetter
from pumpkin_pulse.operator.mesh import Build123DToMesh

def test_material_property_setter():

    # Make operators
    material_property_allocator = MaterialPropertyAllocator()
    sphere_material_property_setter = SphereMaterialPropertySetter(center=(0, 0, 0), radius=5.0)
    mesh_material_property_setter = MeshMaterialPropertySetter()
    build123d_to_mesh = Build123DToMesh()

    # Set material properties
    material_properties = material_property_allocator(
        nr_materials=3,
        origin=(-10, -10, -10),
        spacing=(1, 1, 1),
        shape=(20, 20, 20),
        nr_ghost_cells=1,
    )

    # Make box
    rec = Rectangle(5, 5)
    box = extrude(rec, 5)
    box = Location((5, 5, 5)) * box
    mesh = build123d_to_mesh(box)

    # Set material properties
    material_properties = sphere_material_property_setter(
        material_properties=material_properties,
        id_number=1,
    )
    material_properties = mesh_material_property_setter(
        material_properties=material_properties,
        mesh=mesh,
        id_number=2,
    )

    # Check material properties
    assert material_properties.id.numpy()[10, 10, 10] == 1
    assert material_properties.id.numpy()[15, 15, 16] == 2
