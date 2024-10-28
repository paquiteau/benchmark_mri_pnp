# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: benchmri
#     language: python
#     name: benchmri
# ---

# %%
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from mpl_toolkits.axes_grid1 import ImageGrid
import torch

# %%
import glob 
BENCHMARK = sorted(glob.glob("../outputs/*.parquet"))[-1]
print(BENCHMARK)


# %%
def parse_name(name_str):
    """Parse the `solver_name` of benchopt solver. 
    e.g `"SOLVER_NAME[param1=value1,param2=value2]"` becomes a dict.
    """ 
    # Extract everything within brackets
    params_str = name_str[name_str.find('[')+1:name_str.rfind(']')]
    # Split parameters by comma
    params= dict()
    try:
        params = dict(p.split('=') for p in params_str.split(','))
        params["solver"]=name_str[:name_str.find('[')]
    except ValueError:
        params["solver"]=name_str
    return params


# %%
# BENCHMARK = "../outputs/benchopt_run_2024-10-07_16h21m03.parquet"
BENCHMARK = "../outputs/benchopt_run_2024-10-14_17h47m48.parquet"
AF = 4

# %%
# Define a function to extract parameters
df = pd.read_parquet(BENCHMARK)
# Create a DataFrame from the Series with parsed parameters
df = df.convert_dtypes()
for col in ["version-numpy", "version-scipy", "version-cuda", "benchmark-git-tag","env-OMP_NUM_THREADS", "platform", "platform-architecture", "platform-version", "platform-release", "system-cpus", "system-processor", "system-ram (GB)"]:
    df = df.drop(col, axis=1)
df.columns

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%
# #!python numbers=disable
fig_width_pt = 250   # Get this from LaTeX using \the\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean =  0.5 # 1/1.618       # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height =fig_width*golden_mean       # height in inches
fig_size = [fig_width,fig_height]


# %%
params = {'backend': 'pdf',
          'axes.labelsize': 8,
          'font.size': 8,
          'legend.fontsize': 8,
          'xtick.labelsize': 6,
          'ytick.labelsize': 6,
          'text.usetex': True,
          'figure.figsize': fig_size,
          'mathtext.fontset':'stixsans'}
plt.rcParams.update(params)

plt.rc('text.latex', preamble=r'\def\mathdefault{\mathsf}')

# %%
df["solver_name"].unique()

# %%

# %%
df["display_solver"] = df["solver_name"]+"-"+df["p_solver_iteration"].replace({"classic":"G", "ppnp-cheby":"Cheb", "ppnp-static":"F1"})


# %%
df["iterations"] = df.groupby(["solver_name","p_solver_iteration","p_dataset_seed"],dropna=False)["time"].rank()

# %%
df["p_solver_prior"].unique()

# %%
df["p_solver_prior"] = df["p_solver_prior"].replace({"drunet":"DRUNet", "drunet-denoised":"D-DRUNet", None:"N/A"})
df["solver_name"] = df["solver_name"].apply(lambda x: x.split("[")[0])

df_plot = df[(df["p_solver_prior"] != "DRUNet") & (df["solver_name"].isin(["PNP","HQS"]))].copy()
print(df_plot["p_solver_prior"].unique())
df_plot["p_solver_prior"] = df_plot["p_solver_prior"].replace({"drunet":"DRUNet", "drunet-denoised":"D-DRUNet", None:"N/A"})
df_plot["solver_name"] = df_plot["solver_name"].replace({"FISTA-wavelet":"FISTA-Wavelet", "ncpdnet":"NCPDNET"})
df_plot["p_precond"] = (df_plot["p_solver_iteration"].replace({"classic":"Id", "PGD":"Id", "ppnp-cheby":"Cheb", "ppnp-static":"F1", None:"FISTA"}))

# %%
from matplotlib.offsetbox import (
    VPacker,
    HPacker,
    TextArea,
    AnchoredOffsetbox,
    DrawingArea,
    PaddedBox,
)
from matplotlib.lines import Line2D

from matplotlib.legend import Legend
from matplotlib.patches import FancyBboxPatch, bbox_artist

# %%
sns.set_style("whitegrid")
fig, ax = plt.subplots()
sns.lineplot(df_plot, x="stop_val", y="objective_psnr", style="solver_name", hue="p_precond",ax=ax)
ax.set_xlabel("Iterations")
ax.set_ylabel("PSNR (dB)")
handles, labels = ax.get_legend_handles_labels()
ax.legend_ = None
labels[0] = "Precond."
labels[-3] = "Solver"

precond_handles = handles[1:-3]
precond_labels = labels[1:-3]
solver_handles = handles[-2:]
solver_labels = labels[-2:]

fontsize = 8
handleheight = 1
handlelength = 1.5
legend_handler_map = Legend._default_handler_map
# The approximate height and descent of text. These values are
# only used for plotting the legend handle.
descent = 0.35 * fontsize * (handleheight - 0.5)  # heuristic.
height = fontsize * handleheight
width = handlelength * fontsize

handles_boxes_precond = [
    DrawingArea(width=width, height=height, xdescent=0, ydescent=descent)
    for _ in range(len(precond_handles))
]

handles_boxes_solver = [
    DrawingArea(width=width, height=height, xdescent=0, ydescent=descent)
    for _ in range(len(solver_handles))
]


for h, hl in zip(handles_boxes_precond, precond_handles):
    xdata = [0, 0.5]
    l = Line2D([0, width / 2, width], [descent + 1] * 3, markevery=[1])
    l.set_color(hl.get_color())
    l.set_linestyle(hl.get_linestyle())
    l.set_transform(h.get_transform())
    h.add_artist(l)

for h, hl in zip(handles_boxes_solver, solver_handles):
    xdata = [0, 0.5]
    l = Line2D([0, width / 2, width], [descent + 1] * 3, markevery=[1])
    l.set_color(hl.get_color())
    l.set_linestyle(hl.get_linestyle())
    l.set_transform(h.get_transform())
    h.add_artist(l)


packer = VPacker(
    children=[
        HPacker(
            children=[TextArea("Precond")] + [  
                HPacker(children=[h, TextArea(l)], sep=4)
                for h, l in zip(handles_boxes_precond, precond_labels)
            ],
            sep=8,
            align="right",
        ),
        HPacker(
            children=[TextArea("Solver")] + [  
                HPacker(children=[h, TextArea(l)], sep=4)
                for h, l in zip(handles_boxes_solver, solver_labels)
            ],
            sep=8,
            align="right",
        )
    ],
    sep=2,
)

legend = AnchoredOffsetbox(
    child=packer, loc="lower left", bbox_to_anchor=(0,1), bbox_transform=ax.transAxes
)
legend.patch.set_alpha(0.7)
legend.patch.set_edgecolor("gray")
ax.add_artist(legend)
#ax.legend(h,l, loc="lower right",ncols=5)

ax.set_xlim(0,50)
ax.set_ylim(33.5,38)
# setup ticks
ax.tick_params(axis='both', which='major', pad=0)  # move the tick labels



# %%
fig.savefig(f"convergence{AF}.pdf",bbox_inches="tight", pad_inches=0)

# %%

# %%

# %%

# %%
