import warp as wp


@wp.func
def linear_shape_function(
    x: wp.float32,
    x_ijk: wp.float32,
    spacing: wp.float32,
):
    w = 1.0 - wp.abs((x - x_ijk) / spacing)
    w = wp.max(w, 0.0)
    return w
