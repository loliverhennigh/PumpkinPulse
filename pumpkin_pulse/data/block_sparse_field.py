import warp as wp

def BlockSparseField(dtype):
    if dtype == wp.float32:
        return BlockSparseFieldfloat32
    else:
        raise ValueError('Unknown dtype')

@wp.struct
class BlockSparseFieldfloat32:
    # Id information
    data: wp.array4d(dtype=wp.float32)

    # Grid information
    cardinality: wp.int32
    origin: wp.vec3
    spacing: wp.vec3
    shape: wp.vec3i
    ordering: wp.uint8 # 0: SoA, 1: AoS

    # Block information
    block_start_index: wp.vec3i
    block_shape: wp.vec3i
