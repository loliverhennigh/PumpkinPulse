import warp as wp

from pumpkin_pulse.struct.field import Fielduint8, Fieldfloat32

@wp.struct
class TemperatureField:
    # Temperature field
    temperature: Fieldfloat32
