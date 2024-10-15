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
BENCHMARK_AF4 = "../outputs/benchopt_run_2024-10-07_16h21m03.parquet"
BENCHMARK_AF8 = "../outputs/benchopt_run_2024-10-07_18h01m56.parquet"
BENCHMARK_AF16= "../outputs/benchopt_run_2024-10-09_17h04m04.parquet"


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
# BENCHMARK = "../outputs/benchopt_run_2024-10-06_22h48m18.parquet"
# BENCHMARK = 

# %%
# Define a function to extract parameters
df4 = pd.read_parquet(BENCHMARK_AF4)
df8 = pd.read_parquet(BENCHMARK_AF8)
df16 = pd.read_parquet(BENCHMARK_AF16)

# df8["p_dataset_AF"] = 8 # FIxMEEEE
# df4["p_dataset_AF"] = 4 # FIxMEEEE

df = pd.concat([df4,df8,df16])
# Create a DataFrame from the Series with parsed parameters
df = df.convert_dtypes()
for col in ["version-numpy", "version-scipy", "version-cuda", "benchmark-git-tag","env-OMP_NUM_THREADS", "platform", "platform-architecture", "platform-version", "platform-release", "system-cpus", "system-processor", "system-ram (GB)"]:
    df = df.drop(col, axis=1)
df.columns

# %%
df["p_dataset_AF"].unique()

# %%
#df = df[df["p_solver_prior"] != "drunet"]

# %%
# get the last iteration of every solver

lastiter_idx = df.groupby(["solver_name", "p_dataset_seed", "p_dataset_AF"])["time"].idxmax()
dfli = df.loc[lastiter_idx]
dfli["p_solver_prior"] =dfli["p_solver_prior"].replace({None:"N/A"})
dfli["p_solver_iteration"] =dfli["p_solver_iteration"].replace({None:"N/A"})
dfli["solver_name"] = dfli["solver_name"].apply(lambda x: x.split("[")[0])


# %%
dfli = dfli[~dfli["p_solver_prior"].str.contains("drunet-denoised")]

# %%
dfli["p_solver_prior"].unique()

# %%
dfp = dfli.groupby(["solver_name", "p_solver_prior", "p_solver_iteration", "p_dataset_AF"], 
                   dropna=False)[["time", "objective_psnr", "objective_ssim"]].agg(["mean", "std"])

# %%
dfp

# %%
dfp2 = dfp.reset_index("p_dataset_AF")



# %%
# format the field of the table to be nicer
dfp3 = dfp2.pivot(columns="p_dataset_AF")
dfp3 = dfp3.reset_index()
dfp3["p_solver_prior"] = dfp3["p_solver_prior"].replace({"drunet":"DRUNet", "drunet-denoised":"D-DRUNet", None:"N/A"})
dfp3["solver_name"] = dfp3["solver_name"].apply(lambda x: x.split("[")[0])
dfp3["solver_name"] = dfp3["solver_name"].replace({"FISTA-wavelet":"FISTA-Wavelet", "ncpdnet":"NCPDNET\TblrNote{1}"})
dfp3["p_precond"] = (dfp3["p_solver_iteration"].replace({"classic":"Id", "PGD":"Id", "ppnp-cheby":"Cheb", "ppnp-static":"F1", None:"N/A"}))

# %%
dfp3

# %%
TABLE_TEMPLATE = r"""
[# set rowcounter = namespace(value=3) #]
\begin{longtblr}[note{1} = {{trained for AF=4}},
caption = {[- dataconfig -]},
label ={tab:[- dataconfig -]},
]{
width=\textwidth,
ll[- "r"*2 * (AFS | length)  -],
[# for dd in data #]
[# set blocklength = dd |count #]
cell{[- rowcounter.value -]}{1}={r=[- blocklength -],c=1}{c},
[# set rowcounter.value = rowcounter.value + blocklength  #]
[# endfor #]
cell{1}{3}={c=3}{c},
cell{1}{6}={c=3}{c},
cell{1}{1}={r=2}{c},
cell{1}{2}={r=2}{c},
hline{2}={3-5}{leftpos=-1,rightpos=1, endpos=true},
hline{2}={6-8}{leftpos=-1,rightpos=1, endpos=true},
colsep=3pt,
}
\toprule
\rotatebox{90}{Prec.} & Solver & PSNR  & & & SSIM & &  \\ 
  & Name  [#- for af in AFS #] & AF=[- af -] [#- endfor #] [#- for af in AFS #] & AF=[- af -] [#- endfor #]\\
[# for dd in data #]
\midrule [# for r in dd #] [# if loop.index == 1#] 
\rotatebox{90}{[- r.p_precond -]} [# endif #] 
 & [- r.solver_name -] 
[#- for metrics in ['objective_psnr', 'objective_ssim'] #]
[#- for AF in AFS #]
 & \[- RF[r["_".join([metrics, "rank", AF])]] -][- "{%.3f}" | format(r["_".join([metrics,"mean", AF])]) -]
[#- endfor #][#- endfor #]\\
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
AFS = ["4", "8", "16"]
ddf = dfp3.copy() #dfp3[dfp3["p_solver_prior"] == "DRUNet"].copy()
ddf.columns = ["_".join([str(aa) for aa in a]).rstrip("_") for a in ddf.columns.to_flat_index()]
ddf_ranked = ddf[~ddf["solver_name"].str.contains("NCPDNET")].copy()
for AF in AFS:
    for metric, ascending in [("objective_psnr",False),("objective_ssim",False), ("time", True)]:
        ddf_ranked["_".join([metric, "rank", AF])] = ddf_ranked["_".join([metric, "mean", AF])].rank(ascending=ascending).astype(int)

ddf_ranked = pd.concat([ddf_ranked, ddf[ddf["solver_name"].str.contains("NCPDNET")]])
for name, sddf in ddf_ranked.groupby("p_precond"):
    sddf=sddf.replace({np.nan:None})
    organized.append(sddf.to_dict("records"))


# %%
organized[-1]

# %%
dataconfig = df.reset_index().iloc[0]['data_name']
template = latex_jinja_env.get_template("base_table")
table_str = template.render(data=organized,dataconfig=dataconfig, RF={1: r"textbf",2: r"underline", None:"textit" }| {n:"textrm" for n in range(3,20)}, AFS=AFS)

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
with open(f"table_AF48.tex", "w") as f:
    f.write(table_str)

# %%

# %%

# %%

# %%

# %%

# %%
