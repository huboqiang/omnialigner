"""Visualization utilities; 3D and tensor dependencies are loaded on demand."""
from importlib import import_module as _import_module

__all__ = ["plt", "sns", "pv", "kpt_pairs", "kpt_gradient", "plot_nchw_2d",
           "generte_3d_rgba", "plot_stack_rgba", "plot_nhwc_3d", "bin_to_mesh",
           "imshow_IHC"]

_EXPORTS = {
    "plt": ("matplotlib_init", "plt"), "sns": ("matplotlib_init", "sns"),
    "pv": ("pyvista_3d", "pv"),
    "kpt_pairs": ("keypoint_viz", "plot_kpt_pairs"),
    "kpt_gradient": ("keypoint_viz", "plot_kpts_gradient"),
    "plot_nchw_2d": ("image_viz", "plot_nchw_2d"),
    "generte_3d_rgba": ("image_viz", "generte_3d_rgba"),
    "imshow_IHC": ("omniplot_3d", "apply_colormap_dask"),
    "plot_nhwc_3d": ("pyvista_3d", "plot_nhwc_3d"),
    "plot_stack_rgba": ("pyvista_3d", "stack_rgba_images"),
    "bin_to_mesh": ("pyvista_3d", "bin_to_mesh"),
}
_MODULES = {module for module, _ in _EXPORTS.values()}


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
