# Base class for all operators

from dense_plasma_focus.compute_backend import ComputeBackend

class Operator:
    """
    Base class for all operators
    """

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        """
        Apply the operator to a input. This method will call the
        appropriate apply method based on the compute backend.
        """

    def __repr__(self):
        return f"{self.__class__.__name__}()"
