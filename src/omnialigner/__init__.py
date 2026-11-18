"""OmniAligner public API, with optional components imported on first use."""
from importlib import import_module as _import_module

__all__ = ["config", "align", "pl", "kp", "pp", "tl", "dtypes", "metrics", "external", "integration", "Omni3D", "read_file"]

_EXPORTS = {
    "config": ("configs", None), "align": ("align", None),
    "pl": ("plotting", None), "kp": ("keypoints", None),
    "pp": ("preprocessing", None), "tl": ("utils", None),
    "dtypes": ("dtypes", None), "metrics": ("metrics", None),
    "external": ("external", None), "integration": ("integration", None),
    "Omni3D": ("omni_3D", "Omni3D"), "read_file": ("datasets", "read_file"),
}
# These module attributes were also exposed by the previous eager imports.
_MODULES = {name for name, _ in _EXPORTS.values()} | {"cache_files", "logging", "vendor"}


def __getattr__(name):
    if name in _EXPORTS:
        module_name, attribute = _EXPORTS[name]
    elif name in _MODULES:
        module_name, attribute = name, None
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _import_module(f".{module_name}", __name__)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS) | _MODULES)
