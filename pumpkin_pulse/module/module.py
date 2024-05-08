# Base class for all modules

from dense_plasma_focus.compute_backend import ComputeBackend


class Module:
    """
    Base class for all physics modules
    """

    def __init__(self, state: dict, compute_backend: ComputeBackend):
        self.state = state
        self.compute_backend = compute_backend
