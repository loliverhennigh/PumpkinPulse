from typing import Tuple, Union
from build123d import Compound

from pumpkin_pulse.material import Material


class Solid:
    def __init__(
        self,
        geometry: Compound,
        material: Material,
        label: str = "",
        initial_electric_field: Union[None, Tuple[float, int]] = None,
        permiable: bool = False, # If particles can pass through the solid
    ):
        self.geometry = geometry
        self.material = material
        self.label = label
        self.initial_electric_field = initial_electric_field
        self.permiable = permiable

    @property
    def has_initial_electric_field(self):
        return self.initial_electric_field is not None

    @property
    def bounds(self):
        return self.geometry.bounds

    def __repr__(self):
        return f"Solid({self.geometry}, {self.material})"

    def __str__(self):
        return f"Solid with geometry {self.geometry} and material {self.material}"
