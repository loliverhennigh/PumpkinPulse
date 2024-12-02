import warp as wp
import cupy as cp
#import jax.dlpack as jdlpack
#import jax
 
def _cupy_to_backend(cupy_array, backend):
    # Convert cupy array to backend array
    dl_array = cupy_array.toDlpack()
    if backend == "warp":
        backend_array = wp.from_dlpack(dl_array)
    elif backend == "cupy":
        backend_array = cupy_array
    else:
        raise ValueError(f"Backend {backend} not supported")
    return backend_array


def _backend_to_cupy(backend_array, backend):
    # Convert backend array to cupy array
    if backend == "warp":
        dl_array = wp.to_dlpack(backend_array)
    elif backend == "cupy":
        return backend_array
    else:
        raise ValueError(f"Backend {backend} not supported")
    cupy_array = cp.fromDlpack(dl_array)
    return cupy_array

def _stream_to_backend(stream, backend):
    # Convert stream to backend stream
    if backend == "warp":
        backend_stream = wp.Stream(cuda_stream=stream.ptr)
    elif backend == "cupy":
        backend_stream = stream
    else:
        raise ValueError(f"Backend {backend} not supported")
    return backend_stream


