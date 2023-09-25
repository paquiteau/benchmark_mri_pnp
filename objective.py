from benchopt import BaseObjective, safe_import_context

# Protect import to allow manipulating objective without importing library
# Useful for autocompletion and install commands
with safe_import_context() as import_ctx:
    import numpy as np
    from modopt.math.metrics import psnr, ssim


class Objective(BaseObjective):
    name = "MRI reconstruction"

    install_cmd = 'conda'
    requirements = ['pip:python-pysap']

    # All parameters 'p' defined here are available as 'self.p'
    # parameters = {
    #     'fit_intercept': [False],
    # }
    parameters = {}

    def get_one_result(self):
        # Return one solution. This should be compatible with 'self.compute'.
        return np.zeros(self.image.shape)

    def set_data(self, kspace_data, fourier_op, image):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        self.kspace_data = kspace_data
        self.fourier_op = fourier_op
        self.image = image

    def evaluate_result(self, beta):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        return dict(
            value=-psnr(beta, self.image),
            value_ssim=ssim(beta, self.image),
            value_psnr=psnr(beta, self.image)
        )
        # return psnr(beta, self.image)

    def get_objective(self):
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver.
        # They are customizable.
        return dict(
            kspace_data=self.kspace_data,
            fourier_op=self.fourier_op,
            image=self.image
        )
