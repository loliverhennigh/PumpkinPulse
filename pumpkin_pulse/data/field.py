import warp as wp

def Field(dtype):
    if dtype == wp.float32:
        return Fieldfloat32
    elif dtype == wp.int32:
        return Fieldint32
    elif dtype == wp.uint8:
        return Fielduint8
    else:
        raise ValueError('Unknown dtype')

@wp.struct
class Fieldfloat32:
    # Id information
    data: wp.array4d(dtype=wp.float32)

    # Grid information
    cardinality: wp.int32
    origin: wp.vec3
    spacing: wp.vec3
    shape: wp.vec3i
    offset: wp.vec3i
    ordering: wp.uint8 # 0: SoA, 1: AoS

@wp.struct
class Fieldint32:
    # Id information
    data: wp.array4d(dtype=wp.int32)

    # Grid information
    cardinality: wp.int32
    origin: wp.vec3
    spacing: wp.vec3
    shape: wp.vec3i
    offset: wp.vec3i
    ordering: wp.uint8 # 0: SoA, 1: AoS

@wp.struct
class Fielduint8:
    # Id information
    data: wp.array4d(dtype=wp.uint8)

    # Grid information
    cardinality: wp.int32
    origin: wp.vec3
    spacing: wp.vec3
    shape: wp.vec3i
    offset: wp.vec3i
    ordering: wp.uint8 # 0: SoA, 1: AoS
