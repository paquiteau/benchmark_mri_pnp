#!/usr/bin/env python3
from benchopt import BaseSolver, safe_import_context
from pathlib import Path

with safe_import_context() as import_ctx:
    import os
    from fastmri_recon.models.subclassed_models.ncpdnet import NCPDNet
    import tensorflow as tf
    import numpy as np

    # don't use the full GPU memory
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)


proj_dir = Path(__file__).parent.parent


class Solver(BaseSolver):

    name = "pseudo-inverse"
    install_cmd = "pip"
    sampling_strategy = "run_once"
    requirements = ["deepinv", "mrinufft[gpunufft]"]

    def set_objective(self, kspace_data, physics, trajectory_name):
        # Convert the kspace data from torch to tf
        self.kspace_data = kspace_data
        self.physics = physics

    def run(self, stopval=None):
        # Run the NCPDNet model on the kspace data
        x = self.physics.A_dagger(self.kspace_data)
        self.x_estimate = x

    def get_result(self):
        return {
            "x_estimate": self.x_estimate.numpy().squeeze(),
            "cost": 0,
        }
