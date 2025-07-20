import logging
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import font_manager


warnings.filterwarnings("ignore")
logging.getLogger('fontTools').propagate = False

PY_PATH = "/cluster/home/bqhu_jh/share/miniconda3/envs/cuda1.7/lib/python3.8/site-packages"
MPL_TTY_PATH = "matplotlib/mpl-data/fonts/ttf"
font_files = font_manager.findSystemFonts(
    fontpaths=f"{PY_PATH}/{MPL_TTY_PATH}", fontext='ttf')

if os.path.exists(f'{PY_PATH}/{MPL_TTY_PATH}/Arial.ttf'):
    font_manager.fontManager.addfont(f'{PY_PATH}/{MPL_TTY_PATH}/Arial.ttf')
    font_manager.fontManager.addfont(f'{PY_PATH}/{MPL_TTY_PATH}/Arial Bold.ttf')
    plt.rcParams['font.family'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("white", {"font.family": ["Arial"]})

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
