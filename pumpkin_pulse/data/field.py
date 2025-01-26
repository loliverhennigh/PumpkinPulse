import warp as wp

def Field(dtype):
    if dtype == wp.float32:
        return FieldFloat32
    elif dtype == wp.int32:
        return FieldInt32
    elif dtype == wp.uint8:
        return FieldUint8
    else:
        raise ValueError('Unknown dtype')

@wp.struct
class FieldFloat32:
    # Id information
    data: wp.array4d(dtype=wp.float32)

    # Grid information
    cardinality: wp.int32
    origin: wp.vec3
    spacing: wp.vec3
    shape: wp.vec3i
    offset: wp.vec3i

@wp.struct
class FieldFloat32:
    # Id information
    data: wp.array4d(dtype=wp.float32)

    # Grid information
    cardinality: wp.int32
    origin: wp.vec3
    spacing: wp.vec3
    shape: wp.vec3i
    offset: wp.vec3i

@wp.struct
class FieldInt32:
    # Id information
    data: wp.array4d(dtype=wp.int32)

    # Grid information
    cardinality: wp.int32
    origin: wp.vec3
    spacing: wp.vec3
    shape: wp.vec3i
    offset: wp.vec3i

@wp.struct
class FieldUint8:
    # Id information
    data: wp.array4d(dtype=wp.uint8)

    # Grid information
    cardinality: wp.int32
    origin: wp.vec3
    spacing: wp.vec3
    shape: wp.vec3i
    offset: wp.vec3i
