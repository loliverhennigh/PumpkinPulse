# Base class for voxelizing geometries
import numpy as np
import warp as wp

from pumpkin_pulse.data.field import Fielduint8
from pumpkin_pulse.operator.voxelize.sdf import SignedDistanceFunction
from pumpkin_pulse.operator.voxelize.tube import Tube
from pumpkin_pulse.operator.geometry.geometry import Geometry
from pumpkin_pulse.operator.geometry.circuit.circuit import Circuit

class Coil(Circuit):
    """
    Base class for initializing a coil
    """

    @staticmethod
    def _helix(
        radius: float,
        pitch: float,
        nr_turns: int,
        center: tuple,
        resolution: int = 500,
        flip: bool = False,
    ):
        """
        Create a helix path
        """

        t = np.linspace(0, nr_turns, resolution * nr_turns)
        t = np.linspace(-1.0 / resolution, nr_turns + 1.0 / resolution, resolution * nr_turns)
        x = radius * np.cos(2.0 * np.pi * t) + center[0]
        y = radius * np.sin(2.0 * np.pi * t) + center[1]
        if not flip:
            z = pitch * t + center[2]
        if flip:
            z = -pitch * t + center[2]
        path = np.array([x, y, z]).T
        return path

    @staticmethod
    def _spiral(
        start_radius: float,
        end_radius: float,
        center: tuple,
        resolution: int = 500,
    ):
        """
        Create a spiral path
        """

        # Create a spiral path out of two half circles
        theta_1 = np.linspace(-np.pi / resolution, np.pi + np.pi / resolution, resolution)
        x_1 = (end_radius + start_radius) / 2.0 * np.cos(theta_1) + center[0] - (end_radius - start_radius) / 2.0
        y_1 = (end_radius + start_radius) / 2.0 * np.sin(theta_1) + center[1]
        z_1 = np.zeros_like(x_1) + center[2] + (theta_1 / np.pi) * (end_radius - start_radius) / 1.5
        path_1 = np.array([x_1, y_1, z_1]).T
        theta_2 = np.linspace(np.pi - np.pi / resolution, 2.0 * np.pi + np.pi / resolution, resolution)
        x_2 = end_radius * np.cos(theta_2) + center[0]
        y_2 = end_radius * np.sin(theta_2) + center[1]
        z_2 = np.zeros_like(x_2) + center[2] - ((theta_2 - 2.0 * np.pi) / np.pi) * (end_radius - start_radius) / 1.5
        path_2 = np.array([x_2, y_2, z_2]).T
        path = np.concatenate([path_1, path_2], axis=0)

        #import matplotlib.pyplot as plt
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.plot(path[:, 0], path[:, 1], path[:, 2])
        #plt.show()

        #a = start_radius
        #b = (end_radius - start_radius) / (2.0 * np.pi * nr_turns)
        #theta = np.linspace(0, 2.0 * np.pi * nr_turns, resolution * nr_turns)
        #radius = a + b * theta
        #x = radius * np.cos(theta) + center[0]
        #y = radius * np.sin(theta) + center[1]
        #z = np.zeros_like(x) + center[2]
        #path = np.array([x, y, z]).T
        return path

    def __init__(
        self,
        coil_radius: float,
        cable_radius: float,
        insulator_thickness: float,
        nr_turns_z: int,
        nr_turns_r: int,
        conductor_id: int,
        insulator_id: int,
        center: tuple[float, float, float],
        centeraxis: tuple[float, float, float]=(0.0, 0.0, 1.0),
        angle: float=0.0,
        square: bool=False,
    ):

        # Total cable thickness
        total_cable_thickness = 2.0 * insulator_thickness + 2.0 * cable_radius

        # Create path traced for coil to trace
        path = []

        # Create spiral path inwards
        if nr_turns_r > 1:
            path.append(
                self._spiral(
                    coil_radius + (nr_turns_r - 0.5) * total_cable_thickness,
                    coil_radius + 0.5 * total_cable_thickness,
                    center=[
                        center[0],
                        center[1],
                        center[2] - 0.5 * total_cable_thickness * nr_turns_z - total_cable_thickness,
                    ],
                )
            )
     
        # Create coil paths
        for i in range(nr_turns_r):

            # Create a helix path
            if i == 0:
                path.append(
                    self._helix(
                        coil_radius + (i + 0.5) * total_cable_thickness,
                        total_cable_thickness,
                        nr_turns_z + 1,
                        center=[
                            center[0],
                            center[1],
                            center[2] - 0.5 * total_cable_thickness * nr_turns_z - total_cable_thickness,
                        ],
                        flip=False,
                    )
                )
            elif i % 2 == 0:
                path.append(
                    self._helix(
                        coil_radius + (i + 0.5) * total_cable_thickness,
                        total_cable_thickness,
                        nr_turns_z,
                        center=[
                            center[0],
                            center[1],
                            center[2] - 0.5 * total_cable_thickness * nr_turns_z,
                        ],
                        flip=False,
                    )
                )
            elif i % 2 == 1:
                path.append(
                    self._helix(
                        coil_radius + (i + 0.5) * total_cable_thickness,
                        total_cable_thickness,
                        nr_turns_z,
                        center = [
                            center[0],
                            center[1],
                            center[2] + 0.5 * total_cable_thickness * nr_turns_z,
                        ],
                        flip=True,
                    )
                )

            # Break if last turn
            if i == nr_turns_r - 1:
                break

            # Create spiral path outwards
            if i % 2 == 0:
                path.append(
                    self._spiral(
                        coil_radius + (i + 0.5) * total_cable_thickness,
                        coil_radius + (i + 1.5) * total_cable_thickness,
                        center=[
                            center[0],
                            center[1],
                            center[2] + 0.5 * total_cable_thickness * nr_turns_z,
                        ],

                    )
                )
            else:
                path.append(
                    self._spiral(
                        coil_radius + (i + 0.5) * total_cable_thickness,
                        coil_radius + (i + 1.5) * total_cable_thickness,
                        center=[
                            center[0],
                            center[1],
                            center[2] - 0.5 * total_cable_thickness * nr_turns_z,
                        ],
                    )
                )

        # Concatenate path
        self.path = np.concatenate(path, axis=0)

        # Create radius array for insulator and conductor
        self.conductor_radius = np.full(self.path.shape[0], cable_radius)
        self.insulator_radius = np.full(self.path.shape[0], insulator_thickness + cable_radius)

        # Set ids for conductor and insulator
        self.conductor_id = conductor_id
        self.insulator_id = insulator_id

        # Create operator for tube
        self.tube_operator = Tube()

        # Make input and output points and normals
        input_point = np.array([
            center[0] + coil_radius + (nr_turns_r - 0.5) * total_cable_thickness,
            center[1],
            center[2] - 0.5 * total_cable_thickness * nr_turns_z - total_cable_thickness,
        ])
        input_normal = np.array([0.0, -1.0, 0.0])
        if nr_turns_r % 2 == 0:
            output_point = np.array([
                center[0] + coil_radius + (nr_turns_r - 0.5) * total_cable_thickness,
                center[1],
                center[2] - 0.5 * total_cable_thickness * nr_turns_z,
            ])
        else:
            output_point = np.array([
                center[0] + coil_radius + (nr_turns_r - 0.5) * total_cable_thickness,
                center[1],
                center[2] + 0.5 * total_cable_thickness * nr_turns_z,
            ])
        output_normal = np.array([0.0, 1.0, 0.0])

        # Rotate input and output points and normals
        input_point = Geometry._rotate_point(
            input_point,
            center,
            centeraxis,
            angle,
        )
        input_normal = Geometry._rotate_point(
            input_normal,
            center,
            centeraxis,
            angle,
        )
        output_point = Geometry._rotate_point(
            output_point,
            center,
            centeraxis,
            angle,
        )
        output_normal = Geometry._rotate_point(
            output_normal,
            center,
            centeraxis,
            angle,
        )

        # Square coil
        self.square = square

        # Initialize parent class
        super().__init__(
            input_point,
            input_normal,
            output_point,
            output_normal,
        )


    def __call__(
        self,
        id_field: Fielduint8,
    ):

        # Apply tube operator to path
        self.tube_operator(
            id_field,
            wp.from_numpy(self.path, wp.vec3),
            wp.from_numpy(self.insulator_radius, wp.float32),
            self.insulator_id,
            self.square,
        )
        self.tube_operator(
            id_field,
            wp.from_numpy(self.path, wp.vec3),
            wp.from_numpy(self.conductor_radius, wp.float32),
            self.conductor_id,
            self.square,
        )
        return id_field
