

from dense_plasma_focus.operator.operator import Operator


class Electromagnatism(Voxelizer, TemporalModule):
    """Class for the electromagnatism of the robot"""

    def __init__(
        self,
        compound: Compound,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        shape: Tuple[int, int, int],
        materials: List[Material],
        operators: List[Operator],
        state: Dict[str, Any] = dict(),
        backend: Backend = Backend.WARP
    ):

        # Make state for fdtd simulation
        state['electric_field_x'] = backend.array([s+2 for s in shape], dtype=np.float32)
        state['electric_field_y'] = backend.array([s+2 for s in shape], dtype=np.float32)
        state['electric_field_z'] = backend.array([s+2 for s in shape], dtype=np.float32)
        state['magnetic_field_x'] = backend.array([s+2 for s in shape], dtype=np.float32)
        state['magnetic_field_y'] = backend.array([s+2 for s in shape], dtype=np.float32)
        state['magnetic_field_z'] = backend.array([s+2 for s in shape], dtype=np.float32)
        state['impressed_current_x'] = backend.array([s+2 for s in shape], dtype=np.float32)
        state['impressed_current_y'] = backend.array([s+2 for s in shape], dtype=np.float32)
        state['impressed_current_z'] = backend.array([s+2 for s in shape], dtype=np.float32)

        # Make operator for performing fdtd simulation
        operators['ElectricFieldUpdate'] = ElectricFieldUpdate(backend)
        operators['MagneticFieldUpdate'] = MagneticFieldUpdate(backend)

        # Call super
         super().__init__(
            compound=compound,
            spacing=spacing,
            origin=origin,
            shape=shape,
            materials=materials,
            operators=operators,
            state=state,
            backend=backend
        )

    @property
    def dt(self):
        return 1e-9

    def initialize(self):

        pass

    def step(self):
        self.operator['ElectricFieldUpdate'](
            self.state['electric_field_x'],
            self.state['electric_field_y'],
            self.state['electric_field_z'],
            self.state['impressed_current_x'],
            self.state['impressed_current_y'],
            self.state['impressed_current_z'],
            self.dt
        )
        self.operator['MagneticFieldUpdate'](
            self.state['magnetic_field_x'],
            self.state['magnetic_field_y'],
            self.state['magnetic_field_z'],
            self.state['electric_field_x'],
            self.state['electric_field_y'],
            self.state['electric_field_z'],
            self.dt
        )
