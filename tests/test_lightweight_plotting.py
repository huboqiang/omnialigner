"""Plot-only installs must not import registration or 3D dependencies."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"


class LightweightPlottingTests(unittest.TestCase):
    def run_python(self, script):
        env = dict(os.environ, PYTHONPATH=str(SRC), MPLBACKEND="Agg")
        result = subprocess.run([sys.executable, "-c", textwrap.dedent(script)],
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_bare_import_does_not_load_optional_dependencies(self):
        self.run_python('''
            import sys
            import omnialigner as om
            assert not any(x in sys.modules for x in ("torch", "pyvista", "dask", "matplotlib"))
            assert {"Omni3D", "read_file", "dtypes", "pl", "pp", "tl", "cache_files", "logging", "vendor"} <= set(dir(om))
            try: om.not_a_public_name
            except AttributeError: pass
            else: raise AssertionError("Unknown attributes must fail normally")
        ''')

    def test_existing_api_names_resolve_to_original_objects(self):
        self.run_python('''
            import sys, types
            import omnialigner as om
            root_all = ["config", "align", "pl", "kp", "pp", "tl", "dtypes", "metrics", "external", "integration", "Omni3D", "read_file"]
            assert om.__all__ == root_all and set(root_all) <= set(dir(om))
            aliases = {"config":"configs", "align":"align", "kp":"keypoints", "pp":"preprocessing", "tl":"utils", "dtypes":"dtypes", "metrics":"metrics", "external":"external", "integration":"integration"}
            for alias, name in aliases.items():
                module = types.ModuleType("omnialigner." + name)
                sys.modules[module.__name__] = module
                assert getattr(om, alias) is module
            for name, module_name in [("Omni3D", "omni_3D"), ("read_file", "datasets")]:
                module = types.ModuleType("omnialigner." + module_name)
                sentinel = object(); setattr(module, name, sentinel)
                sys.modules[module.__name__] = module
                assert getattr(om, name) is sentinel
            pl = om.pl
            expected = ["plt", "sns", "pv", "kpt_pairs", "kpt_gradient", "plot_nchw_2d", "generte_3d_rgba", "plot_stack_rgba", "plot_nhwc_3d", "bin_to_mesh", "imshow_IHC"]
            assert pl.__all__ == expected and set(expected) <= set(dir(pl))
            exports = {"matplotlib_init":{"plt":"plt", "sns":"sns"}, "pyvista_3d":{"pv":"pv", "plot_stack_rgba":"stack_rgba_images", "plot_nhwc_3d":"plot_nhwc_3d", "bin_to_mesh":"bin_to_mesh"}, "keypoint_viz":{"kpt_pairs":"plot_kpt_pairs", "kpt_gradient":"plot_kpts_gradient"}, "image_viz":{"plot_nchw_2d":"plot_nchw_2d", "generte_3d_rgba":"generte_3d_rgba"}, "omniplot_3d":{"imshow_IHC":"apply_colormap_dask"}}
            for module_name, names in exports.items():
                module = types.ModuleType("omnialigner.plotting." + module_name)
                for original in names.values(): setattr(module, original, object())
                sys.modules[module.__name__] = module
                for public, original in names.items(): assert getattr(pl, public) is getattr(module, original)
            assert "torch" not in sys.modules and "pyvista" not in sys.modules
        ''')

    def test_real_plot_uses_system_arial_and_true_type_pdf(self):
        self.run_python('''
            import io, sys
            import omnialigner as om
            plt = om.pl.plt
            from matplotlib import font_manager
            assert "torch" not in sys.modules and "pyvista" not in sys.modules
            assert plt.rcParams["pdf.fonttype"] == plt.rcParams["ps.fonttype"] == 42
            assert plt.rcParams["svg.fonttype"] == "none"
            try: arial = font_manager.findfont("Arial", fallback_to_default=False)
            except ValueError: arial = None
            if arial: assert plt.rcParams["font.family"] == ["Arial"]
            else: assert font_manager.findfont(plt.rcParams["font.family"])
            fig, ax = plt.subplots(); text = ax.set_title("Region accuracy 123")
            ax.plot([0, 1], [0, 1]); fig.canvas.draw()
            if arial: assert text.get_fontproperties().get_name() == "Arial"
            pdf = io.BytesIO(); fig.savefig(pdf, format="pdf")
            assert b"/FontFile2" in pdf.getvalue() and b"/Subtype /Type3" not in pdf.getvalue()
            svg = io.BytesIO(); fig.savefig(svg, format="svg")
            assert b"<text" in svg.getvalue()
            plt.close(fig)
        ''')

    def test_missing_arial_falls_back_without_breaking_plotting(self):
        self.run_python('''
            from unittest.mock import patch
            import omnialigner as om
            with patch("matplotlib.font_manager.findfont", side_effect=ValueError("Arial unavailable")):
                plt = om.pl.plt
            assert plt.rcParams["font.family"] == ["DejaVu Sans"]
            fig, ax = plt.subplots(); ax.set_title("Fallback"); fig.canvas.draw(); plt.close(fig)
        ''')

    def test_optional_font_directory_registers_fonts(self):
        self.run_python('''
            import os, shutil, tempfile
            from pathlib import Path
            import matplotlib
            from matplotlib import font_manager
            with tempfile.TemporaryDirectory() as folder:
                # Matplotlib's bundled open-source font, never a proprietary asset.
                target = Path(folder) / "custom.ttf"
                shutil.copy2(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf", target)
                os.environ["OMNIALIGNER_FONT_DIR"] = folder
                import omnialigner as om
                plt = om.pl.plt
                assert any(Path(f.fname).resolve() == target.resolve() for f in font_manager.fontManager.ttflist)
                assert plt.rcParams["pdf.fonttype"] == 42
        ''')


if __name__ == "__main__":
    unittest.main()
