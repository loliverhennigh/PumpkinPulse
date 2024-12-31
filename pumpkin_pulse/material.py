from dataclasses import dataclass
from build123d import Part, Compound
from anytree.search import findall
from typing import Union


# Material class
@dataclass(frozen=True)
class Material:
    name: str
    color: str = "white"
    eps: float = 8.854e-12  # F/m
    mu: float = 4.0 * 3.14159e-7  # H/m
    sigma_e: float = 0.0  # S/m
    sigma_m: float = 0.0  # S/m
    initial_e : Union[float, None] = None

    @property
    def is_conductive(self):
        return self.sigma_e > 0.0 or self.sigma_m > 0.0

    @property
    def constant_conductivity(self):
        return isinstance(self.sigma_e, float) and isinstance(self.sigma_m, float)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


VACUUM = Material(
    name="Vacuum",
    color="white",
    eps=8.854e-12,
    mu=4.0 * 3.14159e-7,
    sigma_e=0.0,
    sigma_m=0.0,
)
COPPER = Material(
    name="Copper",
    color="orange",
    eps=8.854e-12,
    mu=4.0 * 3.14159e-7,
    sigma_e=5.96e7,
    sigma_m=0.0,
)
QUARTZ = Material(
    name="Quartz",
    color="blue",
    eps=8.854e-12 * 3.9,
    mu=4.0 * 3.14159e-7,
    sigma_e=0.0,
    sigma_m=0.0,
)


# function to get all materials in a compound
def get_materials_in_compound(compound: Compound):
    materials = {VACUUM: []}
    for part in findall(compound, filter_=lambda node: isinstance(node, Part)):
        if part.material not in materials:
            materials[part.material] = []
        materials[part.material].append(part)
    return materials
