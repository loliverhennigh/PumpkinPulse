from typing import Tuple, Union, List
from build123d import Compound

from pumpkin_pulse.particle import Particle


class Plasma:
    def __init__(
        self,
        geometry: Compound,
        particles: List[Particle],
        number_density: float,
        micro_to_macro_ratio: float,
        temperature: float, # in Kelvin
    ):
        self.geometry = geometry
        self.particles = particles
        self.number_density = number_density
        self.micro_to_macro_ratio = micro_to_macro_ratio
        self.temperature = temperature

        # Calculate the number of particles in the plasma
        self.volume = Compound.compute_mass(geometry)
        self.nr_real_particles = int(self.number_density * self.volume)
        self.nr_macro_particles = int(self.nr_real_particles / self.micro_to_macro_ratio)

    @property
    def bounds(self):
        return self.geometry.bounds

    def __repr__(self):
        return f"Plasma with geometry {self.geometry} and particles {self.particles}"

    def __str__(self):
        return f"Plasma with geometry {self.geometry} and particles {self.particles}"

