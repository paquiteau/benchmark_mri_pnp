from benchopt import BaseObjective, safe_import_context
import numpy as np
from benchopt.stopping_criterion import SufficientProgressCriterion

# Protect import to allow manipulating objective without importing library
# Useful for autocompletion and install commands
with safe_import_context() as import_ctx:
    from benchmark_utils.metrics import compute_ssim, compute_psnr


class Objective(BaseObjective):
    name = "MRI-reconstruction"

    install_cmd = "conda"
    requirements = ["torch"]
    # All parameters 'p' defined here are available as 'self.p'
    # parameters = {
    #     'fit_intercept': [False],
    # }

    def set_data(self, kspace_data, physics, target):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        self.kspace_data = kspace_data
        self.physics = physics
        self.target = target

    def evaluate_result(self, x_estimate, cost):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        psnr = compute_psnr(x_estimate, self.target)
        ssim = compute_ssim(x_estimate, self.target)
        return dict(
            psnr=psnr,
            ssim=ssim,
            value=psnr,
            cost=cost,
        )

    def save_final_results(self, x_estimate, cost):
        return x_estimate

    def get_one_result(self):
        return {
            "x_estimate": np.zeros(self.physics.nufft.shape, dtype=np.complex64),
            "cost": None,
        }

    def get_objective(self):
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver.
        # They are customizable.

        return dict(kspace_data=self.kspace_data, physics=self.physics)
