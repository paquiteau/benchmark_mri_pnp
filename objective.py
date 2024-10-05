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

    def set_data(self, kspace_data, physics, target, target_denoised, trajectory_name):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        self.kspace_data = kspace_data
        self.physics = physics
        self.target = target
        self.target_denoised = target_denoised
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

    def evaluate_result(self, x_estimate, cost, scale_target=1.0):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        psnr = compute_psnr(self.target * scale_target, x_estimate.squeeze())
        psnr_denoised = compute_psnr(
            self.target_denoised * scale_target, x_estimate.squeeze()
        )
        ssim = compute_ssim(self.target * scale_target, x_estimate.squeeze())
        ssim_denoised = compute_ssim(
            self.target_denoised * scale_target, x_estimate.squeeze()
        )
        return dict(
            psnr=psnr,
            ssim=ssim,
            psnr_denoised=psnr_denoised,
            ssim_denoised=ssim_denoised,
            value=psnr,
            cost=cost,
        )

    def save_final_results(self, x_estimate, cost, scale_target=1.0):
        return (x_estimate / scale_target, self.target, self.target_denoised)

    def get_one_result(self):
        return {
            "x_estimate": np.zeros(self.physics.nufft.shape, dtype=np.complex64),
            "cost": None,
            "scale_target": 1.0,
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
