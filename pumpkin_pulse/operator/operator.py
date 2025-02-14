# Base class for all operators

class Operator:
    """
    Base class for all operators
    """

    def __call__(self, *args, **kwargs):
        """
        Apply the operator to a input. This method will call the
        appropriate apply method based on the compute backend.
        """
