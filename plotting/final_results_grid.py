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
BENCHMARK = "../outputs/benchopt_run_2024-09-09_11h22m24.parquet"

BENCHMARK = "../outputs/benchopt_run_2024-10-06_18h12m12.parquet"


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
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows*3,ncols), axes_pad=0.1,
                     cbar_location="right",
                    cbar_mode="edge",
                    cbar_size="7%",
                    cbar_pad="2%",
                    )
    axes_cols =  [c for col in grid.axes_column for c in col]
    paired_axes = list(zip(axes_cols[::3], axes_cols[1::3], axes_cols[2::3]))
    for (ax, eax, eax2 ), solver_name in zip(paired_axes[1:], df["solver_name"].unique()):
        
        sub_df = df[df["solver_name"] == solver_name]
        args = parse_name(solver_name)
        filter_name = {k:v for k, v in args.items() if k in var_params_names}
        result_file = list(sub_df["final_results"])[0]
        img, target, target_preprocessed = np.load(Path(result_file).resolve(), allow_pickle=True)
        img = abs(img.squeeze())
        vmin = abs(target).min()
        vmax = abs(target).max()
        im_range = ax.imshow(img, vmin=vmin, vmax=vmax, cmap="gray", origin='lower')
        eim_range = eax.imshow(abs(abs(img) - abs(target)),vmin=0, vmax=vmax/20, cmap="inferno", origin='lower')
        eim_range2 = eax2.imshow(abs(abs(img) - abs(target_preprocessed)),vmin=0, vmax=vmax/10, cmap="inferno", origin='lower')
#       
        psnr_max = list(sub_df["objective_psnr"])[-1]
        ssim_max = list(sub_df["objective_ssim"])[-1]
        ax.set_title(f"PSNR={psnr_max:.3f}db, \nSSIM={ssim_max:.3f}")
        ax.axis('off')
        sub_df
        ax.text(2,img.shape[1]-2,"\n".join([f"{k}={v}" for k,v in filter_name.items()]), ha="left", va="top", color="red")
    grid.cbar_axes[0].colorbar(im_range)
    grid.cbar_axes[1].colorbar(eim_range)
    grid.cbar_axes[2].colorbar(eim_range2)

    paired_axes[0][0].imshow(abs(target), vmin=vmin, vmax=vmax,origin="lower", cmap="gray")
    paired_axes[0][0].set_title("Ground Truth")
    paired_axes[0][0].axis("off")
    paired_axes[0][1].axis("off")
    paired_axes[0][2].axis("off")

    return fixed_params, fig
    
def plot_grid_results(df, max_cols=10, figsize=(10,10)):
    # # Create a DataFrame from the Series with parsed parameters
    # df = pd.concat([pd.DataFrame(list(df['solver_name'].apply(parse_name))), df], axis=1)
    # df = df.convert_dtypes()
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
df = pd.read_parquet(BENCHMARK)
print(df.columns)
for name, sub_df in df.groupby("p_solver_prior"):
    sub_df =sub_df.reset_index(drop=True)
    plot_grid_results(sub_df, max_cols=10, figsize=(20,20))

# %%
df = pd.read_parquet(BENCHMARK)
df = df.sort_values("objective_psnr", ascending=False)

last_point = df.groupby("solver_name")["time"].idxmax()
last_point = df.loc[last_point]

ll = last_point.sort_values("objective_psnr", ascending=True)
plot_one_dataset(ll, max_cols=10, figsize=(20,30))

# %%
ll

# %%
ll.columns

# %%
# Table

# %%
TABLE_TEMPLATE = r"""\begin{longtblr}[note{1} = {{trained for AF=4}},
caption = {[- dataconfig -]},
label ={tab:[- dataconfig -]},
remark{Note} = {Some general note. Some general note. Some general note.}]{
width=\textwidth,
llrrrrr,
cell{3}{1}={r=5,c=1}{c},
cell{8}{1}={r=5,c=1}{c},
cell{13}{1}={r=3,c=1}{c},
cell{1}{3}={c=2}{c},
cell{1}{5}={c=2}{c},
hline{2}={3-4}{leftpos=-1,rightpos=-1,endpos=true},
hline{2}={5-6}{rightpos=-1,leftpos=-1,endpos=true},
colsep=3pt,
}
\toprule
D & Solver Name  & PSNR  & & SSIM &    & Time \\ 
         &          & raw  & prep & raw & prep & \\ 
[# for dd in data #]
\midrule [# for r in dd #] [# if loop.index == 1#] \rotatebox{90}{[- r.p_solver_prior -]} [# endif #] 
& [- r.p_solver_iteration -] 
[# for metrics in ['psnr', 'psnr_denoised', 'ssim','ssim_denoised'] #]
& [- "\%s{%.3f}" | format(RF.get(r.get("rank_objective_"+metrics), "textrm"),r.get("objective_"+metrics)) -]
[# endfor #]
& [- "%.3f" | format( r.time) -] \\
[# endfor #]
[# endfor #]
\bottomrule
\end{longtblr}
"""

fulllatex = r"""\documentclass[border=3cm,preview]{standalone}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{subcaption}
\usepackage{booktabs}
\usepackage{tabularray}
\UseTblrLibrary{booktabs}
\begin{document}
[- content -]

\end{document}
"""

# %%

import jinja2
latex_jinja_env = jinja2.Environment(
        block_start_string = '[#',
        block_end_string = '#]',
        variable_start_string = '[-',
        variable_end_string = '-]',
        comment_start_string = '%#',
        comment_end_string = '%#',
        line_statement_prefix = '%%',
        line_comment_prefix = '%#',
        trim_blocks = True,
        autoescape = False,
        loader=jinja2.DictLoader({"base_table": TABLE_TEMPLATE, "standalone":fulllatex})
    )

# %%
ll = ll.sort_values("solver_name")
ll["solver_name"] = ll["solver_name"].apply(lambda x: x.split("[")[0])

ll["p_solver_iteration"]= ll["p_solver_iteration"].replace({"classic":"HQS",
                                                  "FISTA":"PNP-FISTA",
                                                  "PGD": "PNP-PGD",
                                                  "ppnp-cheby": "P$^2$NP-Cheb",
                                                  "ppnp-static": "P$^2$NP-F1",
                                                 })
ll["p_solver_prior"] = ll["p_solver_prior"].replace({
    "drunet":"DRUNet",
    "drunet-denoised":"D-DRUNet", 
    None: "N/A",
})
ll["solver_name"] = ll["solver_name"].replace({"pseudo-inverse":"CG",
                                               "FISTA-wavelet":"FISTA",
                                               "ncpdnet":"NCPDNET\TblrNote{1}",
                                              })

# %%
dataconfig = ll.iloc[0]['data_name']

# %%

organized = []
for _, df in  ll.groupby("p_solver_prior", dropna=False):
    for metric in ["psnr", "ssim", "psnr_denoised", "ssim_denoised"]:
        df[f"rank_objective_{metric}"] = df[f"objective_{metric}"].rank(ascending=False)
    for i, r in df.iterrows():
        if r["p_solver_iteration"] is None:
            df.loc[i, "p_solver_iteration"] = r["solver_name"]
            
    organized.append(df.to_dict("records"))
template = latex_jinja_env.get_template("base_table")
table_str = template.render(data=organized,dataconfig=dataconfig,RF={1: r"textbf",2: r"underline"})

# %%

# %%
# %%
# compile the latex and show it here 
import tempfile
import os
import subprocess 
from IPython.display import IFrame

full_tex = latex_jinja_env.get_template("standalone").render(content = table_str, dataconfig=dataconfig)
with tempfile.TemporaryDirectory() as tmpdir:
    with open(Path(tmpdir) / "tmp.tex", mode="w") as fp:
        fp.write(full_tex)
    subprocess.run(["pdflatex", Path(tmpdir) / "tmp.tex"])
    IFrame(Path(tmpdir)/"tmp.pdf", 600,800)

# %%
with open(f"table_{ll.iloc[0]['data_name']}.tex", "w") as f:
    f.write(table_str)

# %%
print(table_str)

# %%

# %%

# %%
