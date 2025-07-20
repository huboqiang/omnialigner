"""Visualization utilities for OmniAligner.

This module provides a collection of visualization tools for displaying keypoints,
images, and 3D data in various formats.
"""

from .matplotlib_init import plt
from .matplotlib_init import sns
from .pyvista_3d import pv

from .keypoint_viz import plot_kpt_pairs as kpt_pairs
from .keypoint_viz import plot_kpts_gradient as kpt_gradient
from .image_viz import plot_nchw_2d, generte_3d_rgba
from .omniplot_3d import apply_colormap_dask as imshow_IHC
from .pyvista_3d import plot_nhwc_3d
from .pyvista_3d import stack_rgba_images as plot_stack_rgba
from .pyvista_3d import bin_to_mesh

__all__ = ["plt", "sns", "pv", "kpt_pairs", "kpt_gradient", "plot_nchw_2d", 
           "generte_3d_rgba", "plot_stack_rgba", "plot_nhwc_3d", 'bin_to_mesh', 
           "imshow_IHC"]
