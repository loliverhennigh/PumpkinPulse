# Base class for all modules

from dense_plasma_focus.compute_backend import ComputeBackend
from dense_plasma_focus.module.module import Module
from dense_plasma_focus.operator.operator import Operator

class Voxelizer(Module):
    """
    Base class for all physics modules
    """

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

        # Set parameters
        self.compound = compound
        self.spacing = spacing
        self.origin = origin
        self.shape = shape
        self.materials = materials

        # Make operator for performing voxelization
        operators['Voxelize'] = Voxelize(backend)

        # Add material_id to state
        state['material_id'] = backend.array((shape[0], shape[1], shape[2]), dtype=np.uint8)

        # Call super
        super().__init__(operators, state, compute_backend)

    def initialize(self):
        pass

    #def run(self):
    def __call__(self):
        self.voxelize(
            self.state['material_id'],
            self.compound,
            self.spacing,
            self.origin,
            self.shape,
            self.materials,
        )
