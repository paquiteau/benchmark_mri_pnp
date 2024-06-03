from benchopt import BaseObjective, safe_import_context
import numpy as np
from benchopt.stopping_criterion import SufficientProgressCriterion

# Protect import to allow manipulating objective without importing library
# Useful for autocompletion and install commands
with safe_import_context() as import_ctx:
    from benchmark_utils.metrics import compute_psnr, compute_ssim



class Objective(BaseObjective):
    name = "MRI-reconstruction"

    install_cmd = "conda"
    requirements = [
        # "pip:modopt",
        "pip:mri-nufft",
        "pip:cupy",
        "pip:ptwt",
        # "pip:pywavelets",
    ]
    # All parameters 'p' defined here are available as 'self.p'
    # parameters = {
    #     'fit_intercept': [False],
    # }

    def get_one_result(self):
        # Return one solution. This should be compatible with 'self.evaluate_results'.
        return np.zeros(self.target.squeeze(0).shape)

    def set_data(self, kspace_data, kspace_data_hat, target, kspace_mask, images, mask, smaps):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        self.kspace_data = kspace_data
        self.kspace_data_hat = kspace_data_hat
        self.kspace_mask = kspace_mask
        self.target = target
        self.mask = mask
        self.images = images
        self.smaps = smaps

    def evaluate_result(self, x_estimate):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.

        ret_dict = dict(
            value=compute_ssim(self.target, x_estimate, self.mask),
            psnr=compute_psnr(self.target, x_estimate),
        )
        return ret_dict

    def get_objective(self):
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver.
        # They are customizable.

        return dict(
            kspace_data=self.kspace_data, kspace_mask=self.kspace_mask, smaps=self.smaps
        )