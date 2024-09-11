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
#     display_name: bench-mri
#     language: python
#     name: bench-mri
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
#BENCHMARK = "../outputs/benchopt_run_2024-09-09_11h22m24.parquet"

# %%
def fmt_dict(**kwargs):
    """Format a dict in the k=v format."""
    return " ".join([f"{k}={v}" for k,v in filter_name.items()])


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
# Define a function to extract parameters
df = pd.read_parquet(BENCHMARK)
# Create a DataFrame from the Series with parsed parameters
df = pd.concat([pd.DataFrame(list(df['solver_name'].apply(parse_name))), df], axis=1)
df = df.convert_dtypes()
# FIXME: https://github.com/benchopt/benchopt/issues/734
df = df.drop("version-numpy", axis=1)

fixed_columns = df.columns[df.apply(pd.Series.nunique) == 1]
fixed_params = {c: df.loc[0, c] for c in fixed_columns}

df = df.loc[:,df.apply(pd.Series.nunique) != 1]
var_params_names = df.columns[df.apply(pd.Series.nunique) != 1]
df.columns


# %%
def plot_one_dataset(df, max_cols, figsize=(10,10)):
    fig = plt.figure(figsize=figsize,dpi=300)
    n_img = len(df["solver_name"].unique())
    ncols = min(n_img, max_cols) + 1
    nrows = int(np.ceil(n_img / max_cols))
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows*2,ncols), axes_pad=0.2,
                     cbar_location="right",
                    cbar_mode="edge",
                    cbar_size="7%",
                    cbar_pad="2%",
                    )
    axes_cols =  [c for col in grid.axes_column for c in col]
    paired_axes = list(zip(axes_cols[::2], axes_cols[1::2]))
    for (ax, eax), solver_name in zip(paired_axes[1:], df["solver_name"].unique()):
        
        sub_df = df[df["solver_name"] == solver_name]
        args = parse_name(solver_name)
        filter_name = {k:v for k, v in args.items() if k in var_params_names}
        result_file = list(sub_df["final_results"])[0]
        img, target = np.load(Path(result_file).resolve(), allow_pickle=True)
        img = abs(img.squeeze())
        vmin = abs(target).min()
        vmax = abs(target).max()
        im_range = ax.imshow(img, vmin=vmin, vmax=vmax, cmap="gray", origin='lower')
        eim_range = eax.imshow(abs(abs(img) - abs(target)),vmin=0, vmax=vmax/20, cmap="inferno", origin='lower')
        psnr_max = list(sub_df["objective_psnr"])[-1]
        ssim_max = list(sub_df["objective_ssim"])[-1]
        ax.set_title(f"PSNR={psnr_max:.3f}db, \nSSIM={ssim_max:.3f}")
        ax.axis('off')
        sub_df
        ax.text(2,img.shape[1]-2,"\n".join([f"{k}={v}" for k,v in filter_name.items()]), ha="left", va="top", color="red")
    grid.cbar_axes[0].colorbar(im_range)
    grid.cbar_axes[1].colorbar(eim_range)
    paired_axes[0][0].imshow(abs(target), vmin=vmin, vmax=vmax,origin="lower", cmap="gray")
    paired_axes[0][0].set_title("Ground Truth")
    paired_axes[0][1].axis("off")
    return fixed_params, fig
    
def plot_grid_results(benchmark_file, max_cols=10, figsize=(10,10)):
    df = pd.read_parquet(benchmark_file)
    
    # Create a DataFrame from the Series with parsed parameters
    df = pd.concat([pd.DataFrame(list(df['solver_name'].apply(parse_name))), df], axis=1)
    df = df.convert_dtypes()
    # FIXME: https://github.com/benchopt/benchopt/issues/734
    df = df.drop("version-numpy", axis=1)
    
    fixed_columns = df.columns[df.apply(pd.Series.nunique) == 1]
    fixed_params = {c: df.loc[0, c] for c in fixed_columns}
    
    df = df.loc[:,df.apply(pd.Series.nunique) != 1]
    var_params_names = df.columns[df.apply(pd.Series.nunique) != 1]

    if "data_name" in df.columns:
        for data_name in df["data_name"].unique():
            df_data = df[df["data_name"] == data_name]
            plot_one_dataset(df_data, max_cols=max_cols, figsize=figsize)
    else:    
        plot_one_dataset(df, max_cols=max_cols, figsize=figsize)


# %%
plot_grid_results(BENCHMARK, max_cols=10, figsize=(20,20))

# %%
import seaborn as sns
sns.set_theme(style="white", palette=None)


# %%
df2 = df.reindex(None)
df2

# %%
df_filter = df[~df["solver_name"].str.contains("iteration=FISTA")]
sns.relplot(data=df_filter, x="stop_val", y="objective_ssim", hue="solver_name", kind="line", label="solver")

# %%

# %%
