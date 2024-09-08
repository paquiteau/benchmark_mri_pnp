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
#     display_name: bench
#     language: python
#     name: bench
# ---

# %%
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from mpl_toolkits.axes_grid1 import ImageGrid
import torch

# %%
BENCHMARK = "../outputs/benchopt_run_2024-09-08_15h47m55.csv"


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
    params = dict(p.split('=') for p in params_str.split(','))
    params["solver"]=name_str[:name_str.find('[')]
    return params


# %%
# Define a function to extract parameters
df = pd.read_csv(BENCHMARK)
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

def plot_grid_results(benchmark_file, max_cols=10, figsize=(10,10)):
    df = pd.read_csv(benchmark_file)
    
    # Create a DataFrame from the Series with parsed parameters
    df = pd.concat([pd.DataFrame(list(df['solver_name'].apply(parse_name))), df], axis=1)
    df = df.convert_dtypes()
    # FIXME: https://github.com/benchopt/benchopt/issues/734
    df = df.drop("version-numpy", axis=1)
    
    fixed_columns = df.columns[df.apply(pd.Series.nunique) == 1]
    fixed_params = {c: df.loc[0, c] for c in fixed_columns}
    
    df = df.loc[:,df.apply(pd.Series.nunique) != 1]
    var_params_names = df.columns[df.apply(pd.Series.nunique) != 1]
    
    fig = plt.figure(figsize=figsize)
    n_img = len(df["solver_name"].unique())
    ncols = min(n_img, max_cols)
    nrows = (n_img // max_cols) + 1
    
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows,ncols), axes_pad=0.2)
    
    for ax, solver_name in zip(grid, df["solver_name"].unique()):
        
        sub_df = df[df["solver_name"] == solver_name]
        args = parse_name(solver_name)
        filter_name = {k:v for k, v in args.items() if k in var_params_names}
        result_file = list(sub_df["final_results"])[0]
        img = np.load(Path(result_file).resolve(), allow_pickle=True)
        img = img.squeeze(0).squeeze(0).abs().numpy()
        ax.imshow(img, cmap="gray", origin='lower')
        psnr_max = sub_df["objective_psnr"].max()
        ax.set_title(f"PSNR={psnr_max:.3}db")
        ax.axis('off')
        sub_df
        ax.text(2,img.shape[1]-2,"\n".join([f"{k}={v}" for k,v in filter_name.items()]), ha="left", va="top", color="red")
    return fixed_params


# %%
plot_grid_results(BENCHMARK, max_cols=4, figsize=(20,20))

# %%

# %%
