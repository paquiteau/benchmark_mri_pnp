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
from mrinufft.trajectories import display_2D_trajectory
from mrinufft.trajectories import initialize_2D_spiral
import matplotlib.pyplot as plt

# %%
AF = 8
samples_loc = initialize_2D_spiral(
    int(320 / AF), 320, nb_revolutions=1, in_out=True
)


# %%

# # #!python numbers=disable
fig_width_pt = 250   # Get this from LaTeX using \the\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean =  1 # 1/1.618       # Aesthetic ratio
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
fig, ax = plt.subplots()

from mrinufft.trajectories.display import displayConfig

displayConfig.fontsize=9

ax= display_2D_trajectory(samples_loc,one_shot=True, subfigure=ax)
fig.savefig(f"spiral_{AF}.pdf", bbox_inches="tight", pad_inches=0)

# %%

# %%
