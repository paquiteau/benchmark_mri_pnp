from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import pysap
    import numpy as np
    from mri.operators import WaveletN
    from mri.reconstructors import SingleChannelReconstructor

    from modopt.opt.proximity import SparseThreshold
    from modopt.opt.linear import Identity


class Solver(BaseSolver):
    """Zero order solution"""
    name = 'fista'

    install_cmd = 'conda'
    requirements = ['pip:python-pysap']

    # any parameter defined here is accessible as a class attribute
    parameters = {
        "wavelet_name": ["sym8"],
        "nb_scales": [4],
        "lambd": [2 * 1e-7]
    }

    def set_objective(
        self,
        kspace_data,
        fourier_op,
        image,
        wavelet_name="sym8",
        nb_scales=4,
        lambd=2 * 1e-7
    ):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.kspace_data = kspace_data
        self.fourier_op = fourier_op
        self.image = image
        self.wavelet_name = wavelet_name
        self.nb_scales = nb_scales
        self.lambd = lambd

    def run(self, n_iter):
        linear_op = WaveletN(wavelet_name="sym8", nb_scales=4)
        regularizer_op = SparseThreshold(Identity(), 2 * 1e-7,
                                         thresh_type="soft")
        reconstructor = SingleChannelReconstructor(
            fourier_op=self.fourier_op,
            linear_op=linear_op,
            regularizer_op=regularizer_op,
            gradient_formulation='synthesis',
            verbose=0
        )
        x_final, _, _ = reconstructor.reconstruct(
            kspace_data=self.kspace_data,
            optimization_alg="fista",
            num_iterations=max(n_iter, 1)
        )

        self.w = pysap.Image(data=np.abs(x_final))

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.w
