from benchopt import BaseSolver, safe_import_context

#from benchmark_utils.stopping_criterion import ModoptCriterion
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    from modopt.base.backend import get_backend
    from modopt.opt.algorithms import POGM, ForwardBackward

    from modopt.math.metrics import psnr, ssim, mse
    from modopt.opt.linear import WaveletTransform
    from modopt.base.backend import get_backend
    from mrinufft import get_operator

    from benchmark_utils.sure_prox import AutoWeightedSparseThreshold
    from benchmark_utils.gradients import GradSynthesis


class Solver(BaseSolver):
    """Zero order solution"""

    name = "forward-backward"

    install_cmd = "conda"
    sampling_strategy = "callback"
    requirements = [
        "pip:modopt",
        "pip:mri-nufft",
        "pip:cupy",
        "pip:ptwt",
        "pip:pywavelets",
    ]

    grad_formulation = "synthesis"
    # any parameter defined here is accessible as a class attribute
    parameters = {
        "optimizer": ["POGM"],
        "init": ["zero"],
        "wavelet_name": ["db4-3"],
        "backend": ["numpy", "cupy"],
        "nufft": ["gpunufft", "cufinufft", "stacked-cufinufft", "stacked-gpunufft"],
    }

    stopping_criterion = SufficientProgressCriterion(patience=30)

    def set_objective(
        self,
        kspace_data,
        kspace_mask,
        smaps,
    ):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.



        xp, _ = get_backend(self.backend)
        wname, level = self.wavelet_name.split("-")
        self.linear_op = WaveletTransform(
            wavelet_name=wname,
            shape=smaps.shape[1:],
            mode="zero",
            level=int(level),
            compute_backend=self.backend,
        )

        _ = self.linear_op.op(xp.zeros_like(smaps[0]))

        self.prox_op = AutoWeightedSparseThreshold(
            self.linear_op.coeffs_shape,
            sigma_range="global",
            threshold_estimation="hybrid-sure",
            thresh_range="global",
            threshold_scaler=0.3,
        )
        density = None

        extra_kwargs = dict()
        if "stacked" in self.nufft:
            density = "voronoi"
            extra_kwargs["z_index"] = "auto"
        if  "gpunufft" in self.nufft:
            density = "pipe"
        if self.nufft == "stacked-gpunufft" and self.backend == "cupy":
            smaps = xp.array(smaps)
            extra_kwargs["use_gpu_direct"] = True

        if self.nufft == "gpunufft" and self.backend == "cupy":
            extra_kwargs["use_gpu_direct"] = True

        nufft = get_operator(self.nufft)(
            kspace_mask,
            shape=smaps.shape[1:],
            n_coils=smaps.shape[0],
            smaps=smaps,
            density=density,
            squeeze_dims=True,
            **extra_kwargs,
        )
        if self.backend == "cupy":
            kspace_data = xp.array(kspace_data)
        adj = nufft.adj_op(kspace_data)
        alpha_init = self.linear_op.op(adj)
        self.prox_op.op(alpha_init)

        self.grad_op = GradSynthesis(
            self.linear_op,
            fourier_op=nufft,
            dtype="complex64",
            compute_backend=self.backend,
            input_data_writeable=True,
            verbose=False,
        )

        self.grad_op._obs_data = xp.array(kspace_data)


        opt_kwargs = dict(
            grad=self.grad_op,
            prox=self.prox_op,
            cost=None,
            linear=self.linear_op,
            beta=self.grad_op.inv_spec_rad,
            auto_iterate=False,
            compute_backend=self.backend,
        )
        if self.optimizer == "POGM":
            self.solver = POGM(alpha_init, alpha_init, alpha_init, alpha_init, **opt_kwargs)
        elif self.optimizer == "ForwardBackward":
            self.solver = ForwardBackward(alpha_init, **opt_kwargs)

    def run(self, callback):
        self.x = self.solver._x_old
        while callback():
            self.solver._update()
            self.solver.idx += 1
            self.x = self.solver._x_new

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        x_estimate = self.grad_op.linear_op.adj_op(self.x)
        if self.backend == "cupy":
            x_estimate = x_estimate.get()

        return {
            "alpha_estimate": self.x,
            "x_estimate": x_estimate,
            "cost": self.grad_op.cost(self.x) + self.prox_op.cost(self.x),
        }
