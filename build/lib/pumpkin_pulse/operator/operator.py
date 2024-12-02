# Base class for all operators

from pumpkin_pulse.compute_backend import ComputeBackend


class Operator:
    """
    Base class for all operators
    """

    def __init__(self):
        self.make_kernels()

    def make_kernels(self):
        """
        Make kernels for the operator
        """
        pass

    def __call__(self, *args, **kwargs):
        """
        Apply the operator to a input. This method will call the
        appropriate apply method based on the compute backend.
        """
