"""Portable Matplotlib defaults for editable manuscript figures."""
import os
from pathlib import Path as _Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager


def _register_optional_fonts():
    # Normal system fonts are discovered by Matplotlib itself. Extra fonts are
    # read only when explicitly supplied; no font assets are bundled/downloaded.
    for directory in os.environ.get("OMNIALIGNER_FONT_DIR", "").split(os.pathsep):
        if not directory:
            continue
        folder = _Path(os.path.expandvars(directory)).expanduser()
        if not folder.is_dir():
            continue
        for filename in font_manager.findSystemFonts(fontpaths=[str(folder)], fontext="ttf"):
            try:
                font_manager.fontManager.addfont(filename)
            except (OSError, ValueError, RuntimeError):
                # A malformed optional font must not prevent scoring/plotting.
                continue


_register_optional_fonts()
try:
    font_manager.findfont("Arial", fallback_to_default=False)
    _family = ["Arial"]
except ValueError:
    _family = ["DejaVu Sans"]

sns.set_style("white", {"font.family": _family})
plt.rcParams.update({
    "font.family": _family,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})
