from benchopt import BaseObjective, safe_import_context
import numpy as np
from benchopt.stopping_criterion import SufficientProgressCriterion

# Protect import to allow manipulating objective without importing library
# Useful for autocompletion and install commands
with safe_import_context() as import_ctx:
    from modopt.math.metrics import psnr, ssim, mse
    from modopt.opt.linear import WaveletTransform
    from modopt.base.backend import get_backend
    from mrinufft import get_operator

    from benchmark_utils.sure_prox import AutoWeightedSparseThreshold
    from benchmark_utils.gradients import GradSynthesis


class Objective(BaseObjective):
    name = "MRI-reconstruction"

    install_cmd = "conda"
    requirements = [
        "pip:modopt",
        "pip:mri-nufft",
        "pip:cupy",
        "pip:ptwt",
        "pip:pywavelets",
    ]
    # All parameters 'p' defined here are available as 'self.p'
    # parameters = {
    #     'fit_intercept': [False],
    # }

    def get_one_result(self):
        # Return one solution. This should be compatible with 'self.evaluate_results'.
        xp, _ = get_backend(self.backend)
        return xp.zeros(self.image.shape)

    def set_data(self, kspace_data, kspace_mask, image, smaps):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        self.kspace_data = kspace_data
        self.kspace_mask = kspace_mask
        self.image = image
        self.smaps = smaps

    def evaluate_result(self, alpha_estimate, x_estimate, cost):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        ret_dict = dict(
            value=cost,
            psnr=psnr(x_estimate, self.image),
        )
        return ret_dict

    def get_objective(self):
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver.
        # They are customizable.

        return dict(
            kspace_data=self.kspace_data, kspace_mask=self.kspace_mask, smaps=self.smaps
        )
