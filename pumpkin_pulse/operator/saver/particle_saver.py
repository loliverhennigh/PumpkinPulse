import os
import numpy as np
import pyvista as pv

from pumpkin_pulse.data.particles import Particles
from pumpkin_pulse.operator.operator import Operator

class ParticleSaver(Operator):

    def __call__(
        self,
        particles: Particles,
        filename: str,
        save_velocity: bool = False,
        save_index: bool = False,
    ):

        # Save particles
        nr_particles = particles.nr_particles.numpy()[0]
        position = particles.position[:nr_particles].numpy()
        momentum = particles.momentum[:nr_particles].numpy()
        grid = pv.PolyData(position)
        if save_velocity:
            grid["particle_momentum"] = momentum
        if save_index:
            grid["particle_index"] = np.arange(nr_particles)
        grid.save(filename)

        return None
