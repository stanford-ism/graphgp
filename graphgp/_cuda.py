import jax

try:
    import graphgp_cuda as _graphgp_cuda

    has_cuda_extension = True
except ImportError:
    _graphgp_cuda = None
    has_cuda_extension = False


graphgp_cuda = _graphgp_cuda


def require_cuda_runtime() -> None:
    if not has_cuda_extension:
        raise ImportError("CUDA extension not installed, cannot use cuda=True.")

    try:
        devices = jax.devices("cuda")
    except Exception as exc:
        raise RuntimeError(
            "cuda=True requires a working JAX CUDA backend, but JAX could not initialize one."
        ) from exc

    if not devices:
        raise RuntimeError("cuda=True requires at least one CUDA device, but JAX did not detect any.")
