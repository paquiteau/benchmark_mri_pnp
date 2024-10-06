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
BENCHMARK = "../outputs/benchopt_run_2024-10-06_22h48m18.parquet"

# %%
# Define a function to extract parameters
df = pd.read_parquet(BENCHMARK)
# Create a DataFrame from the Series with parsed parameters
df = df.convert_dtypes()
for col in ["version-numpy", "version-scipy", "version-cuda", "benchmark-git-tag","env-OMP_NUM_THREADS", "platform", "platform-architecture", "platform-version", "platform-release", "system-cpus", "system-processor", "system-ram (GB)"]:
    df = df.drop(col, axis=1)
df.columns

# %%
# get the last iteration of every solver

lastiter_idx = df.groupby(["solver_name", "p_dataset_id"])["time"].idxmax()
dfli = df.loc[lastiter_idx]

# %%
dfp = dfli.groupby(["solver_name", "p_solver_prior", "p_solver_iteration"], as_index=False)[["time", "objective_psnr", "objective_ssim"]].agg(["mean", "std"])
dfp

# %%
# format the field of the table to be nicer

dfp["p_solver_prior"] = dfp["p_solver_prior"].replace({"drunet":"DRUNet", "drunet-denoised":"D-DRUNet"})
dfp["solver_name"] = dfp["solver_name"].apply(lambda x: x.split("[")[0])
dfp["display_solver"] = dfp["solver_name"]+"-"+dfp["p_solver_iteration"].replace({"classic":"G", "ppnp-cheby":"Cheb", "ppnp-static":"F1"})
dfp

# %%

# %%

# %%
TABLE_TEMPLATE = r"""
[# set rowcounter = namespace(value=2) #]
\begin{longtblr}[note{1} = {{trained for AF=4}},
caption = {[- dataconfig -]},
label ={tab:[- dataconfig -]},
remark{Note} = {Some general note. Some general note. Some general note.}]{
width=\textwidth,
llrrr,
[# for dd in data #]
[# set blocklength = dd |count #]
cell{[- rowcounter.value -]}{1}={r=[- blocklength -],c=1}{c},
[# set rowcounter.value = rowcounter.value + blocklength  #]
[# endfor #]
colsep=3pt,
}
\toprule
D & Solver Name & PSNR  & SSIM & Time \\ 
[# for dd in data #]
\midrule [# for r in dd #] [# if loop.index == 1#] 
\rotatebox{90}{[- r.p_solver_prior -]} [# endif #] 
 & [- r.display_solver -] 
[#- for metrics in ['objective_psnr', 'objective_ssim', 'time'] #]
 & \[- RF[r[metrics+"_rank"]] -][- "{%.3f}" | format(r[metrics+"_mean"]) -]
[#- endfor #]\\
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
plop="""
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
organized = []
for _, ddf in  dfp.groupby("p_solver_prior", dropna=False):
    ddf.columns = ["_".join(a).rstrip("_") for a in ddf.columns.to_flat_index()]
    for metric, ascending in [("objective_psnr",False),("objective_ssim",False), ("time", True)]:
        ddf[f"{metric}_rank"] = ddf[f"{metric}_mean"].rank(ascending=ascending).astype(int)
    organized.append(ddf.to_dict("records"))


# %%
organized[0][0]

# %%
dataconfig = df.reset_index().iloc[0]['data_name']
template = latex_jinja_env.get_template("base_table")
table_str = template.render(data=organized,dataconfig=dataconfig,RF={1: r"textbf",2: r"underline"} | {n:"textrm" for n in range(3,20)})

# %%
df.columns

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
with open(f"table_{dataconfig}.tex", "w") as f:
    f.write(table_str)

# %%

# %%

# %%

# %%

# %%
