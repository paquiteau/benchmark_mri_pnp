from benchopt import BaseObjective, safe_import_context
import numpy as np
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim


class Objective(BaseObjective):
    name = "MRI-reconstruction"

    install_cmd = "conda"
    requirements = ["torch"]
    # All parameters 'p' defined here are available as 'self.p'
    # parameters = {
    #     'fit_intercept': [False],
    # }

    def set_data(self, kspace_data, physics, target, trajectory_name):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        self.kspace_data = kspace_data
        self.physics = physics
        self.target = target
        self.trajectory_name = trajectory_name

    def get_objective(self):
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver.
        # They are customizable.

        return dict(
            kspace_data=self.kspace_data,
            physics=self.physics,
            trajectory_name=self.trajectory_name,
        )

    def evaluate_result(self, x_estimate, cost):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        psnr = compute_psnr(self.target, x_estimate.squeeze())
        ssim = compute_ssim(self.target, x_estimate.squeeze())
        return dict(
            psnr=psnr,
            ssim=ssim,
            value=psnr,
            cost=cost,
        )

    def save_final_results(self, x_estimate, cost):
        return (x_estimate, self.target)

    def get_one_result(self):
        return {
            "x_estimate": np.zeros(self.physics.nufft.shape, dtype=np.complex64),
            "cost": None,
        }


def compute_psnr(gt, pred):
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    return compare_psnr(abs(gt), abs(pred), data_range=abs(gt).max() - abs(gt).min())


def compute_ssim(gt, pred):
    """Compute Structural Similarity Index Metric (SSIM)."""
    return compare_ssim(
        abs(gt),
        abs(pred),
        # gt.transpose(1, 2, 0),
        # pred.transpose(1, 2, 0),
        # multichannel=True,
        data_range=abs(gt).max() - abs(gt).min(),
    )
