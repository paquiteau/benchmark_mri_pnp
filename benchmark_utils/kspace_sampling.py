from mrinufft import get_density, get_operator, check_backend
from mrinufft.trajectories import initialize_2D_radial, initialize_2D_spiral, initialize_2D_cones, initialize_2D_sinusoide, initialize_2D_propeller, initialize_2D_rings, initialize_2D_rosette, initialize_2D_polar_lissajous, initialize_2D_lissajous, initialize_2D_waves

from mrinufft.io.nsp import read_trajectory
from mrinufft.trajectories.display import display_2D_trajectory
import numpy as np
from snkf.handlers.acquisition.cartesian_sampling import get_cartesian_mask
import pdb

def get_samples(string, *args, **kwargs):
    
    trajectory_initializers = {
        "sparkling": (initialize_2D_sparkling, ("sparkling_2d",), {}),
        "cartesian": (initialize_2D_cartesian, (args), {"Ns": kwargs['Ns']}),
        "radial": (initialize_2D_radial, (args), kwargs),
        "spiral": (initialize_2D_spiral, (kwargs['Nc'], kwargs['Ns']), {}),
        "cones": (initialize_2D_cones, (kwargs['Nc'], kwargs['Ns']), {"nb_zigzags": 5, "width": 1}),
        "sinusoide": (initialize_2D_sinusoide, (kwargs['Nc'], kwargs['Ns']), {}),
        "propeller": (initialize_2D_propeller, (kwargs['Nc'], kwargs['Ns']), {"nb_strips": 10}),
        "rings": (initialize_2D_rings, (kwargs['Nc'], kwargs['Ns']), {"nb_rings": kwargs['Nc']}),
        "rosette": (initialize_2D_rosette, (kwargs['Nc'], kwargs['Ns']), {}),
        "polar_lissajous": (initialize_2D_polar_lissajous, (kwargs['Nc'], kwargs['Ns']), {}),
        "lissajous": (initialize_2D_lissajous, (kwargs['Nc'], kwargs['Ns']), {}),
        "waves": (initialize_2D_waves, (kwargs['Nc'], kwargs['Ns']), {}),
    }

    if string in trajectory_initializers:
        initializer, args, kwargs_extra = trajectory_initializers[string]
        trajectory = initializer(*args, **kwargs_extra)
        return trajectory.astype(np.float32)
    else:
        raise ValueError("Unsupported trajectory type: {}".format(string))

def initialize_2D_sparkling(path):
    
    traj, params = read_trajectory(path)
    traj *= 2   # The above trajectory is in [-0.25, 0.25] but we want it in [-0.5, 0.5]
    traj = traj
    return traj

def initialize_2D_cartesian(Ns, KMAX = 0.5):

    s = 320
    samples_loc = get_cartesian_mask((s,s), 1, accel_axis = 1)
    Nc = (samples_loc[0,0] == 1).sum()
    loc = np.arange(-s // 2, s // 2) / s
    loc_ = loc[samples_loc[0,0] == 1]
    trajectory = np.zeros((Nc, Ns, 2))
    segment = np.linspace(-1, 1, Ns) * KMAX
    for i in range(0,Nc):
        trajectory[i,:,0] = np.ones(Ns) * loc_[i]
        trajectory[i,:,1] = segment
        
    return trajectory.astype(np.float32)