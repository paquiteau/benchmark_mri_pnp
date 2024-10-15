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
import matplotlib
import pandas as pd
from pathlib import Path
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colorbar 
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
BENCHMARK = "../outputs/benchopt_run_2024-10-07_16h21m03.parquet"
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
fig_width_pt = 500   # Get this from LaTeX using \the\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean =  0.5 # 1/1.618       # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height =fig_width*golden_mean       # height in inches
fig_size = [fig_width,fig_height]


# %%
params = {'backend': 'notebook',
          'axes.labelsize': 6,
          'font.size': 6,
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
# Create a DataFrame from the Series with parsed parameters

lastiter_idx = df.groupby(["solver_name", "p_dataset_id", "data_name", "p_dataset_seed", "p_dataset_AF","p_dataset_id"])["time"].idxmax()

# %%
dfli = df.loc[lastiter_idx]


# %%
dfli = dfli[dfli["p_solver_prior"] != "drunet"]
dfli = dfli[dfli["p_dataset_seed"] == 5]
dfli

# %%
dfli["p_solver_prior"] = dfli["p_solver_prior"].replace({"drunet":"DRUNet", "drunet-denoised":"D-DRUNet", None:"N/A"})
dfli["solver_name"] = dfli["solver_name"].apply(lambda x: x.split("[")[0])
df_plot = dfli.copy()
#df_plot = dfli[(dfli["p_solver_prior"] != "DRUNet") & (df["solver_name"].isin(["PNP","HQS"]))].copy()
print(df_plot["p_solver_prior"].unique())
df_plot["p_solver_prior"] = df_plot["p_solver_prior"].replace({"drunet":"DRUNet", "drunet-denoised":"D-DRUNet", None:"N/A"})
df_plot["solver_name"] = df_plot["solver_name"].replace({"FISTA-wavelet":"FISTA-Wavelet", "ncpdnet":"NCPDNET"})
df_plot["p_precond"] = (df_plot["p_solver_iteration"].replace({"classic":"Id", "PGD":"Id", "ppnp-cheby":"Cheb", "ppnp-static":"F1", None:"FISTA"}))


# %%
df_plot

# %%
df = df_plot
fig = plt.figure(figsize=fig_size,dpi=300)
n_img = len(df_plot)
ncols = n_img + 1
nrows = 1
grid = ImageGrid(fig, 111, nrows_ncols=(nrows*2,ncols), axes_pad=0.01,
                 cbar_location="right",
                cbar_mode=None, # edge
                cbar_size="7%",
                cbar_pad="2%",
                )
axes_cols =  [c for col in grid.axes_column for c in col]
paired_axes = list(zip(axes_cols[::2], axes_cols[1::2]))
for (ax, eax), (_, row) in zip(paired_axes[1:], df.iterrows()):
    #filter_name = {k:v for k, v in args.items() if k in var_params_names}
    result_file = row["final_results"]
    img, target, target_preprocessed = np.load(Path(result_file).resolve(), allow_pickle=True)
    img = abs(img.squeeze())
    vmin = abs(target).min()
    vmax = abs(target).max()
    im_range = ax.imshow(img, vmin=vmin, vmax=vmax, cmap="gray", origin='lower')
    eim_range = eax.imshow(abs(abs(img) - abs(target)),vmin=0, vmax=vmax/15, cmap="inferno", origin='lower')    
    psnr_max = row["objective_psnr"]
    ssim_max = row["objective_ssim"]
    ax.text(0.02,0.98, f"PSNR={psnr_max:.3f}dB\nSSIM={ssim_max:.3f}", color="white",fontsize=4, ha="left", va="top", transform=ax.transAxes,)
    ax.axis('off')
    eax.axis('off')
    ax.text(0.5,1.05, f"{row['solver_name']}-{row['p_precond']}", color="black",fontsize=6, ha="center", va="bottom", transform=ax.transAxes,)



# cbar = grid.cbar_axes[0].colorbar(im_range)
# cbar.formatter.set_powerlimits((0, 0))
# cbar2 = grid.cbar_axes[1].colorbar(eim_range)
# cbar2.formatter.set_powerlimits((0, 0))
# print(cbar.ax.yaxis.get_major_formatter().)
# print(cbar.ax.get_yminorticklabels())
ax = grid.axes_row[0][0]
ax.imshow(abs(target), vmin=vmin, vmax=vmax,origin="lower", cmap="gray")
ax.text(0.5,1.05,"Ground Truth", color="black", fontsize=6, ha="center", va="bottom",transform=ax.transAxes)

paired_axes[0][0].axis("off")
br= paired_axes[0][1] 
br.imshow(np.zeros_like(abs(target))*np.NaN)
br.axis("off")
cax1 = inset_axes(br, 
    width="5%",  # width: 5% of parent_bbox width
    height="95%",  # height: 50%
    loc="center left",
    bbox_to_anchor=(0.45, 0.0, 1, 1),
    bbox_transform=br.transAxes,
    borderpad=0)

cax2 = inset_axes(br, 
    width="5%",  # width: 5% of parent_bbox width
    height="95%",  # height: 50%
    loc="center left",
    bbox_to_anchor=(0.95, 0.0, 1, 1),
    bbox_transform=br.transAxes,
    borderpad=0,)
cax1.imshow(np.linspace(vmin, vmax,50)[:,None], vmin=vmin, vmax=vmax, origin="lower", extent=[0,1, vmin, vmax], aspect="auto", cmap="gray")
cax2.imshow(np.linspace(0, vmax/15,50)[:,None], vmin=0, vmax=vmax/15, origin="lower", extent=[0,1, 0, vmax/15],aspect="auto", cmap="inferno")
cax2.set_xticks([])
cax1.set_xticks([])
exp1 = int(np.floor(np.log10(vmax)))
exp2 = int(np.floor(np.log10(vmax/15)))

cax1.set_yticks([0, 2e-4, 4e-4, vmax], ["0", "2", "4", f"{vmax/(10**exp1):.1f}$\\times$10$^{{{exp1}}}$"], fontsize=3)
cax2.set_yticks([0, 1e-5, 2e-5, 3e-5, vmax/15], ["0", "1", "2", "3",f"{(vmax/15)/(10**exp2):.1f}$\\times$ 10$^{{{exp2}}}$"], fontsize=3)


print(cax2.get_yticks(), vmax/15)
#cax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
# cax1.get_yaxis().get_offset_text().set_visible(False)
# cax2.get_yaxis().get_offset_text().set_visible(False)

# print(cax1.get_yticks())
# cax1.set_yticks(cax1.get_yticks()[1:],["0"]+ [f"{t:.0e}" for t in np.linspace(vmin, vmax, 6)[1:]], fontsize=4)
# cax2.set_yticks(cax2.get_yticks()[1:],["0"]+ [f"{t:.0e}" for t in np.linspace(0, vmax/20, 6)[1:]], fontsize=4)
for ca in [cax1, cax2]:
    # change all spines
    for side in ['top','bottom','left','right']:
        ca.spines[side].set_linewidth(0.1)
        # increase tick width
        ca.tick_params(width=0.1, length=0.5, pad=1)


#fig.colorbar(im_range,cax=cax1, panchor=)
#paired_axes[0][1].colorbar(eim_range)


fig.savefig(f"grid_image{AF}.pdf",bbox_inches="tight", pad_inches=0)


# %%

# %%
