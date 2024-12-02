# Capacitor Circuit 

import os
import numpy as np
import warp as wp
import matplotlib.pyplot as plt
import dataclasses
import itertools
from tqdm import tqdm

wp.init()

from pumpkin_pulse.constructor import Constructor
from pumpkin_pulse.data import Fielduint8, Fieldfloat32
from pumpkin_pulse.operator import Operator
from pumpkin_pulse.operator.voxelize import (
    Tube,
)
from pumpkin_pulse.operator.saver import FieldSaver


class CapacitorCircuit(Operator):

    def __init__(
        self,
        coil_radius: float,
        cable_thickness_r: float,
        cable_thickness_z: float,
        insulator_thickness: float,

    ):

        # Make list of all edges
        x_edges = []
        y_edges = []

        # Make outer outline of coil
        outer_radius = coil_radius + cable_thickness_r
        outer_distance_from_axis = 2.0 * cable_thickness_r + insulator_thickness / 2.0
        outer_arch_angle = np.arcsin(outer_distance_from_axis / outer_radius)
        outer_arch_angle_linspace = np.linspace(outer_arch_angle, 2.0 * np.pi - outer_arch_angle, 100)
        outer_x = outer_radius * np.cos(outer_arch_angle_linspace)
        outer_y = outer_radius * np.sin(outer_arch_angle_linspace)
        x_edges.append(outer_x)
        y_edges.append(outer_y)

        # Make inner outline of coil
        inner_radius = coil_radius - cable_thickness_r
        inner_distance_from_axis = insulator_thickness / 2.0
        inner_arch_angle = np.arcsin(inner_distance_from_axis / inner_radius)
        inner_arch_angle_linspace = np.linspace(inner_arch_angle, 2.0 * np.pi - inner_arch_angle, 100)
        inner_x = inner_radius * np.cos(inner_arch_angle_linspace)
        inner_y = inner_radius * np.sin(inner_arch_angle_linspace)

        # Make outer line outwards
        start_x = outer_x[-1]
        end_x = outer_x[-1] + 2.0 * cable_thickness_z
        start_y = outer_y[-1]
        end_y = outer_y[-1]
        x_edges.append(np.linspace(start_x, end_x, 2))
        y_edges.append(np.linspace(start_y, end_y, 2))

        # Next section
        start_x = end_x
        end_x = start_x
        start_y = end_y
        end_y = start_y + 2.0 * cable_thickness_r
        x_edges.append(np.linspace(start_x, end_x, 2))
        y_edges.append(np.linspace(start_y, end_y, 2))

        # Next section
        start_x = end_x
        end_x = inner_x[-1]
        start_y = end_y
        end_y = inner_y[-1]
        x_edges.append(np.linspace(start_x, end_x, 2))
        y_edges.append(np.linspace(start_y, end_y, 2))

        # Add inner coil
        x_edges.append(np.flip(inner_x))
        y_edges.append(np.flip(inner_y))

        # Next section
        start_x = inner_x[0]
        end_x = outer_x[-1] + 2.0 * cable_thickness_z
        start_y = inner_y[0]
        end_y = inner_y[0]
        x_edges.append(np.linspace(start_x, end_x, 2))
        y_edges.append(np.linspace(start_y, end_y, 2))

        # Next section
        start_x = end_x
        end_x = start_x
        start_y = end_y
        end_y = outer_y[0]
        x_edges.append(np.linspace(start_x, end_x, 2))
        y_edges.append(np.linspace(start_y, end_y, 2))

        # Next section
        start_x = end_x
        end_x = outer_x[0]
        start_y = end_y
        end_y = outer_y[0]
        x_edges.append(np.linspace(start_x, end_x, 2))
        y_edges.append(np.linspace(start_y, end_y, 2))

        # Plot the coil
        x_edges = np.concatenate(x_edges)
        y_edges = np.concatenate(y_edges)
        plt.plot(x_edges, y_edges)
        plt.show()


if __name__ == "__main__":

    # Define the coil parameters
    coil_radius = 1.0
    cable_thickness_r = 0.1
    cable_thickness_z = 0.1
    insulator_thickness = 0.1

    # Create the capacitor circuit
    circuit = CapacitorCircuit(
        coil_radius=coil_radius,
        cable_thickness_r=cable_thickness_r,
        cable_thickness_z=cable_thickness_z,
        insulator_thickness=insulator_thickness,
    )
