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
BENCHMARK = "../outputs/benchopt_run_2024-06-04_10h52m24.parquet"


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
    
    fig = plt.figure(figsize=figsize)
    max_cols = 10
    n_img = len(df["solver_name"].unique())
    ncols = min(n_img, max_cols)
    nrows = (n_img // max_cols) + 1
    
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows,ncols), axes_pad=0.1)
    
    for ax, solver_name in zip(grid, df["solver_name"].unique()):
        
        sub_df = df[df["solver_name"] == solver_name]
        args = parse_name(solver_name)
        filter_name = {k:v for k, v in args.items() if k in var_params_names}
        result_file = "../"+list(sub_df["final_results"])[0]
        img = np.load(Path(result_file).resolve(), allow_pickle=True)
        img = img.squeeze(0).squeeze(0).abs().numpy()
        ax.imshow(img, cmap="gray", origin='lower')
        ax.axis('off')
        sub_df
        ax.set_title(" ".join([f"{k}={v}" for k,v in filter_name.items()]))
    return fixed_params


# %%
plot_grid_results(BENCHMARK)


# %%

def get_result_table(benchmark_file, *metrics):
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
    results = []
    
    for  solver_name in df["solver_name"].unique():
        
        sub_df = df[df["solver_name"] == solver_name]
        args = parse_name(solver_name)
        filter_name = {k:v for k, v in args.items() if k in var_params_names}
        max_time = sub_df["time"].max()
        last_iter = sub_df["time"].idxmax()
        results.append(
            {"run time":max_time} |
            filter_name |     
            {m:sub_df.loc[last_iter, f"objective_{m}"] for m in metrics})
        
    return pd.DataFrame(results)


# %%
get_result_table(BENCHMARK, "psnr", "ssim")

# %%
