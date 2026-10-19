"""Small launch-metadata cache; no tensor/pointer ownership is retained."""

import triton

_CACHE = {}
_VERSION = tuple(map(int, triton.__version__.split(".")[:2]))


def launch(fn, grid, args, **options):
    while not isinstance(fn, triton.runtime.JITFunction):
        fn = fn.fn
    sig = tuple(
        (a.dtype, a.device.index, a.data_ptr() % 16)
        if hasattr(a, "data_ptr")
        else (type(a), a)
        for a in args
    )
    key = (fn, sig, tuple(sorted(options.items())))
    kernel = _CACHE.get(key)
    if kernel is None:
        kernel = fn[grid](*args, **options)
        _CACHE[key] = kernel
    else:
        values = (
            args
            if _VERSION[0] == 3 and 3 <= _VERSION[1] <= 6
            else tuple(a for a, p in zip(args, fn.params) if not p.is_constexpr)
        )
        kernel[(grid + (1, 1))[:3]](*values)
    return kernel
