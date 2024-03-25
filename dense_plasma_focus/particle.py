from dataclasses import dataclass
from build123d import Part, Compound
from anytree.search import findall

# Fundamental constants
ELECTRON_CHARGE = -1.602176634e-19 # C
ELECTRON_MASS = 9.10938356e-31 # kg
PROTON_CHARGE = 1.602176634e-19 # C
PROTON_MASS = 1.6726219e-27 # kg

# Material class
@dataclass(frozen=True)
class Particle:
    name: str
    color: str
    charge: float
    mass: float

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

class Electron(Particle):
    def __init__(self, micro_to_macro_ratio: 1.0e10):
        super().__init__(
            name="Electron",
            color="blue",
            charge=ELECTRON_CHARGE * micro_to_macro_ratio,
            mass=ELECTRON_MASS * micro_to_macro_ratio,
        )

class Proton(Particle):
    def __init__(self, micro_to_macro_ratio: 1.0e10):
        super().__init__(
            name="Proton",
            color="red",
            charge=PROTON_CHARGE * micro_to_macro_ratio,
            mass=PROTON_MASS * micro_to_macro_ratio,
        )
