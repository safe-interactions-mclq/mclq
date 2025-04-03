import types as pytypes
from numba import njit
import inspect

def _force_jit_func(func, seen=None):
    if seen is None:
        seen = {}
    elif func in seen:
        return seen[func]
    elif hasattr(func, "py_func"):
        return func

    orig_globals = func.__globals__
    new_globals = orig_globals.copy()

    for name, obj in orig_globals.items():
        if isinstance(obj, pytypes.FunctionType) and name in func.__code__.co_names:
            new_globals[name] = _force_jit_func(obj, seen)

    cloned_func = pytypes.FunctionType(
        func.__code__,
        new_globals,
        func.__name__,
        func.__defaults__,
        func.__closure__,
    )

    jitted_func = njit(cloned_func)
    seen[func] = jitted_func
    return jitted_func
