from dataclasses import dataclass
from build123d import Part, Compound
from anytree.search import findall

# Fundamental constants
ELECTRON_CHARGE = -1.602176634e-19  # C
ELECTRON_MASS = 9.10938356e-31  # kg
PROTON_CHARGE = 1.602176634e-19  # C
PROTON_MASS = 1.6726219e-27  # kg


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

ELECTRON = Particle(
    name="Electron",
    color="blue",
    charge=ELECTRON_CHARGE,
    mass=ELECTRON_MASS,
)
PROTON = Particle(
    name="Proton",
    color="red",
    charge=PROTON_CHARGE,
    mass=PROTON_MASS,
)
