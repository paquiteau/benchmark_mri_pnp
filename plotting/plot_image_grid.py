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
#BENCHMARK = "../outputs/benchopt_run_2024-10-07_16h21m03.parquet"
#AF = 4

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
params = {#'backend': 'notebook',
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
dfli["p_solver_prior"].unique()


# %%
dfli = df.loc[lastiter_idx]
dfli = dfli[dfli["p_solver_prior"].isin(["drunet-denoised", pd.NA])]
dfli = dfli[dfli["p_dataset_seed"] == 1]
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


df_plot = df_plot[df_plot["p_solver_iteration"].isin(["classic", "ppnp-static", "PGD", pd.NA])]
df_plot


# %%
from mrinufft.trajectories import initialize_2D_spiral, display_2D_trajectory, displayConfig

spiral4 = initialize_2D_spiral(320//4,320, in_out=True)
spiral16 = initialize_2D_spiral(320//16,320, in_out=True)


# %%
def display_traj(trajectory, ax, one_shot=True):
    colors = displayConfig.get_colorlist()
    Nc, Ns = trajectory.shape[:2]
    for i in range(Nc):
        ax.plot(
            trajectory[i, :, 0],
            trajectory[i, :, 1],
            color=colors[i % displayConfig.nb_colors],
            linewidth=0.5,
        )

    # Display one shot in particular if requested
    if one_shot is not False:  # If True or int
        # Select shot
        shot_id = Nc // 2
        if one_shot is not True:  # If int
            shot_id = one_shot

        # Highlight the shot in black
        ax.plot(
            trajectory[shot_id, :, 0],
            trajectory[shot_id, :, 1],
            color=displayConfig.one_shot_color,
            linewidth=0.5,
        )


# %%
ROI_SIZE = [0.1 ,0.1]
ROI_ANCHOR = [0.6,0.5]
def add_inset(ax, img, vmin, vmax, cmap, loc=(0.7,0.7,0.3,0.3)):
    img_shape = img.shape
    # INSET 
    axins = ax.inset_axes(loc)
    axins.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
    
    axins.set_xlim(img_shape[0]*(ROI_ANCHOR[0]), img_shape[0]*(ROI_ANCHOR[0]+ROI_SIZE[0])) 
    axins.set_ylim(img_shape[1]*(ROI_ANCHOR[1]), img_shape[1]*(ROI_ANCHOR[1]+ROI_SIZE[1]))
    for spine in axins.spines.values():
        spine.set(color="lime", linewidth=0.5)
    axins.grid(False)
    axins.set_xticks([])
    axins.set_yticks([])
    ax.indicate_inset((img_shape[0]*ROI_ANCHOR[0], 
                      img_shape[1]*ROI_ANCHOR[1],
                      img_shape[0]*ROI_SIZE[0], 
                      img_shape[1]*ROI_SIZE[1]), edgecolor="lime", alpha=1, linewidth=0.5)

#dfp = df_plot.sort_values(["solver_name", "p_dataset_AF"])
dfp =df_plot
fig = plt.figure(figsize=fig_size,dpi=300)
grid = ImageGrid(fig, 111, nrows_ncols=(2,7), cbar_mode=None, axes_pad=0.01)

[ax.axis("off") for ax in grid]

for (af, dfpp), row_axes in  zip(dfp.groupby("p_dataset_AF"), grid.axes_row):
    for ax, (_, r) in zip(row_axes[:-1], dfpp.iterrows()):
        result_file = r["final_results"]
        img, target, target_preprocessed = np.load(Path(result_file).resolve(), allow_pickle=True)
        img = abs(img.squeeze())
        err = abs(img - abs(target))
        vmin = abs(target).min()
        vmax = abs(target).max()
        im_range = ax.imshow(img, vmin=vmin, vmax=vmax, cmap="gray", origin="lower")
        add_inset(ax, img, vmin, vmax, cmap="gray")
        add_inset(ax, err, 0, vmax/10, cmap="inferno", loc=(0.7,0.,0.3,0.3) )
        psnr_max = r["objective_psnr"]
        ssim_max = r["objective_ssim"]
        ax.text(0.02,0.98, f"PSNR={psnr_max:.3f}dB\nSSIM={ssim_max:.3f}", color="white",fontsize=4, ha="left", va="top", transform=ax.transAxes,)
        label = f"{r['solver_name']}-{r['p_precond']}"
        label=label.replace("-FISTA","")
        ax.text(0.5,1.05, label, color="black",fontsize=6, ha="center", va="bottom", transform=ax.transAxes,)

gt_ax = grid.axes_column[-1][0]
gt_im = gt_ax.imshow(abs(target), vmin=vmin, vmax=vmax, cmap="gray", origin="lower")
add_inset(gt_ax, abs(target), vmin, vmax, cmap="gray")

gt_ax.text(0.5,1.05,"Ground Truth",ha="center", va="bottom", transform=gt_ax.transAxes)


for ax, af in zip(grid.axes_column[0], [4,16]):
    ax.text(-0.05,0.5, f"AF={af}", rotation=90, va="center", ha="center",transform=ax.transAxes)

traj_ax= grid.axes_column[-1][-1].inset_axes((0.1,0.1,0.8,0.8))

# traj_ax.set_xlim(-0.5,0.5)
# traj_ax.set_ylim(-0.5,0.5)
print(traj_ax.axis("on"))
display_traj(spiral16*380+190, traj_ax)
traj_ax.set_xticks([0,190,380],[-0.5,0,0.5], fontsize=4)
traj_ax.yaxis.tick_right()
traj_ax.set_yticks([0,190,380],[-0.5,0,0.5], fontsize=4)
traj_ax.tick_params(width=0.2,length=1,pad=0.1)
traj_ax.set_title("AF=16", pad=3,fontsize=4)
for spine in traj_ax.spines.values():
    spine.set(linewidth=0.5)
traj_ax.grid(linewidth=0.5)

fig.savefig(f"grid_image.pdf",bbox_inches="tight", pad_inches=0)

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
    im_range = ax.imshow(img, vmin=vmin, vmax=vmax, cmap="gray")
    eim_range = eax.imshow(abs(abs(img) - abs(target)),vmin=0, vmax=vmax/15, origin="lower", cmap="inferno")    
    psnr_max = row["objective_psnr"]
    ssim_max = row["objective_ssim"]
    ax.text(0.02,0.98, f"PSNR={psnr_max:.3f}dB\nSSIM={ssim_max:.3f}", color="white",fontsize=4, ha="left", va="top", transform=ax.transAxes,)
    ax.axis('off')
    eax.axis('off')
    ax.text(0.5,1.05, f"{row['solver_name']}-{row['p_precond']}", color="black",fontsize=6, ha="center", va="bottom", transform=ax.transAxes,)



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

fig.show()
fig.savefig(f"grid_image_full.pdf",bbox_inches="tight", pad_inches=0)


# %%

# %%

# %%

# %%
