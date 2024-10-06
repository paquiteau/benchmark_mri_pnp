from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion
import os
from pathlib import Path
import numpy as np
from benchmark_utils.precond import Precond


with safe_import_context() as import_ctx:
    import torch
    from deepinv.models import WaveletDictDenoiser
    from deepinv.optim.data_fidelity import L2
    from deepinv.optim import optim_builder, PnP, Prior, BaseOptim
    from deepinv.optim.optim_iterators.hqs import fStepHQS, gStepHQS
    from deepinv.optim.optim_iterators import OptimIterator, fStep, gStep

    from deepinv.optim.utils import gradient_descent

    from benchmark_utils.utils import stand
    from benchmark_utils.drunet import DRUNet

proj_dir = Path(__file__).parent.parent
DRUNET_PATH = os.environ.get("DRUNET_PATH", proj_dir / "drunet.tar")
DRUNET_DENOISE_PATH = os.environ.get(
    "DRUNET_DENOISE_PATH", proj_dir / "drunet_denoised.tar"
)


class Solver(BaseSolver):
    """HQS with PNP."""

    name = "HQS"

    install_cmd = "conda"
    sampling_strategy = "callback"
    requirements = ["deepinv", "mrinufft[cufinufft]"]
    parameters = {
        "iteration": ["classic", "ppnp-cheby", "ppnp-static"],
        "prior": ["drunet", "drunet-denoised"],
        "sigma": [0.0007],
        "xi": [0.84],
        "lamb": [6.5],
        "max_iter": [20],
        "stepsize": [1.5],
    }
    stopping_criterion = SufficientProgressCriterion(patience=30)

    def skip(self, kspace_data, physics, trajectory_name):
        if self.prior == "drunet" and not os.path.exists(DRUNET_PATH):
            return True, "DRUNet weights not found"
        if self.prior == "drunet-denoised" and not os.path.exists(DRUNET_DENOISE_PATH):
            return True, "DRUNet denoised weights not found"
        return False, ""

    def get_next(self, stop_val):
        return stop_val + 1

    def set_objective(
        self,
        kspace_data,
        physics,
        trajectory_name,
    ):

        # # HACK add the noise here (avoid recomputing smaps and operator for each trial)

        # self.kspace_data = (
        #     kspace_data + torch.randn_like(kspace_data) * self.noise_variance
        # )
        self.kspace_data = kspace_data
        self.physics = physics
        kwargs_optim = dict()
        denoiser = load_drunet(
            DRUNET_DENOISE_PATH if "denoised" in self.prior else DRUNET_PATH
        )
        cpx_denoiser = Denoiser(denoiser)
        prior = PnP(cpx_denoiser)
        kwargs_optim["params_algo"] = get_DPIR_params(
            sigma=self.sigma,
            xi=self.xi,
            lamb=self.lamb,
            stepsize=self.stepsize / physics.nufft.get_lipschitz_cst(),
            n_iter=self.max_iter,
        )

        if "ppnp" in self.iteration:
            _, precond = self.iteration.split("-")
            kwargs_optim["custom_init"] = get_custom_init_ppnp
            iterator = PreconditionedHQSIteration(
                precond=Precond(precond),
            )
        else:
            iterator = "HQS"
            kwargs_optim["custom_init"] = get_custom_init
        self.algo = optim_builder(
            iteration=iterator,
            prior=prior,
            data_fidelity=L2(),
            early_stop=False,
            max_iter=self.max_iter,
            verbose=False,
            **kwargs_optim,
        )

    def run(self, callback):
        with torch.no_grad():
            x_cur = self.algo.fixed_point.init_iterate_fn(
                self.kspace_data, self.physics
            )
            itr = 0
            self.x_estimate, self.cost = self._get_estimate(x_cur)
            while callback():
                x_cur = self.algo.fixed_point.single_iteration(
                    x_cur,
                    itr,
                    self.kspace_data,
                    self.physics,
                    compute_metrics=False,
                    x_gt=None,
                )
                if self.algo.fixed_point.check_iteration:
                    itr += 1
                    self.x_estimate, self.cost = self._get_estimate(x_cur)
                if itr >= self.max_iter:
                    break
        del self.algo

    def get_result(self):
        """Get values to pass to objective."""
        return {
            "x_estimate": self.x_estimate.detach().cpu().numpy(),
            "cost": self.cost,
        }

    def _get_estimate(self, x_cur):
        x_est = x_cur["est"]
        if isinstance(x_est, tuple):
            x_est = x_est[1]
        return x_est, x_cur["cost"]


class Denoiser(torch.nn.Module):
    """Denoiser to wrap DRUNET for Complex data."""

    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    def forward(self, x, sigma, norm=True):
        x = torch.permute(torch.view_as_real(x.squeeze(0)), (0, 3, 1, 2)).to("cuda")
        if norm:
            x = x * 1e4
        x_ = torch.permute(self.denoiser(x, sigma).to("cpu"), (0, 2, 3, 1))
        if norm:
            x_ = x_ * 1e-4
        return torch.view_as_complex(x_.contiguous()).unsqueeze(0)


def load_drunet(path_weights):
    model = DRUNet(in_channels=2, out_channels=2, pretrained=None).to("cuda")
    checkpoint = torch.load(
        path_weights, map_location=lambda storage, loc: storage, weights_only=True
    )
    if "state_dict" in checkpoint.keys():
        checkpoint = checkpoint["state_dict"]
    new_checkpoint = {}
    for key, value in checkpoint.items():
        new_key = key.replace("backbone_net.", "")
        new_checkpoint[new_key] = value
    model.load_state_dict(new_checkpoint)
    model.eval()
    return model


def get_custom_init(y, physics):

    est = physics.A_dagger(y)
    est = torch.zeros_like(est)
    return {"est": (est, est.detach().clone())}


def get_custom_init_ppnp(y, physics):

    est = physics.A_dagger(y)
    est = torch.zeros_like(est)
    # return {
    #     "est": (torch.zeros_like(est), torch.zeros_like(est), torch.zeros_like(est))
    # }
    return {"est": (est, est.detach().clone(), torch.zeros_like(est))}


def get_DPIR_params(sigma=1, xi=1, lamb=2, stepsize=1, n_iter=10):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    # sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), n_iter).astype(np.float32)
    # sigma_denoiser = np.linspace(s1, s2, n_iter).astype(np.float32)
    # stepsize = (sigma_denoiser / max(1e-6, s2)) ** 2
    sigma_denoiser = (sigma * xi ** np.arange(n_iter)).astype(np.float32)
    stepsize = np.ones_like(sigma_denoiser) * stepsize
    # return {"lambda": lamb, "g_param": list(sigma_denoiser), "stepsize": list(stepsize)}
    return {"lambda": lamb, "g_param": list(sigma_denoiser), "stepsize": list(stepsize)}


class PreconditionedHQSIteration(OptimIterator):
    r"""
    Single iteration of half-quadratic splitting.

    Class for a single iteration of the Half-Quadratic Splitting (HQS) algorithm for minimising :math:` f(x) + \lambda g(x)`.
    The iteration is given by


    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k} &= \operatorname{prox}_{\gamma f}(x_k) \\
        x_{k+1} &= \operatorname{prox}_{\sigma \lambda g}(u_k).
        \end{aligned}
        \end{equation*}


    where :math:`\gamma` and :math:`\sigma` are step-sizes. Note that this algorithm does not converge to
    a minimizer of :math:`f(x) + \lambda  g(x)`, but instead to a minimizer of
    :math:`\gamma\, ^1f+\sigma \lambda g`, where :math:`^1f` denotes
    the Moreau envelope of :math:`f`

    """

    def __init__(self, precond=None, **kwargs):
        super(PreconditionedHQSIteration, self).__init__(**kwargs)
        self.g_step = gStepHQS(**kwargs)
        self.f_step = fStepHQSPrecond(**kwargs)
        self.requires_prox_g = True
        self.precond = precond

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics):
        r"""
        General form of a single iteration of splitting algorithms for minimizing :math:`F =  f + \lambda g`, alternating
        between a step on :math:`f` and a step on :math:`g`.
        The primal and dual variables as well as the estimated cost at the current iterate are stored in a dictionary
        $X$ of the form `{'est': (x,z), 'cost': F}`.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the observation.
        :return: Dictionary `{"est": (x, z), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x_prev = X["est"][0]
        if not self.g_first:
            z = self.f_step(
                x_prev, cur_data_fidelity, cur_params, y, physics, precond=self.precond
            )
            z = z.cfloat()  # disgusting but for some reason the above casts to double
            x = self.g_step(z, cur_prior, cur_params)
        #         else:
        #             Not implemented
        x = self.relaxation_step(x, x_prev, cur_params["beta"])
        F = (
            self.F_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost
            else None
        )
        return {"est": (x, z), "cost": F}


class fStepHQSPrecond(fStep):
    r"""
    HQS fStep module.
    """

    def __init__(self, **kwargs):
        super(fStepHQSPrecond, self).__init__(**kwargs)
        self.metric = L2()

    def forward(self, x, cur_data_fidelity, cur_params, y, physics, precond=None):
        r"""
        Single proximal step on the data-fidelity term :math:`f`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
        """
        # in the standard case we would return this: (useful for debugging)
        # return cur_data_fidelity.prox(x, y, physics, gamma=cur_params["stepsize"])
        # instead we do that:
        return self.prox_l2_metric(
            x, y, physics, gamma=cur_params["stepsize"], precond=precond
        )

    def prox_l2_metric(
        self,
        x,
        y,
        physics,
        gamma,
        precond,
        stepsize_inter=1.0,
        max_iter_inter=50,
        tol_inter=1e-3,
    ):
        r"""
        Computes proximal operator of :math:`f(x)=\frac{\gamma}{2}\|Ax-y\|^2`
        in an efficient manner leveraging the singular vector decomposition.

        :param torch.Tensor, float z: signal tensor
        :param torch.Tensor y: measurements tensor
        :param float gamma: hyperparameter :math:`\gamma` of the proximal operator
        :return: (torch.Tensor) estimated signal tensor

        """
        grad = lambda z: gamma * self.metric.grad(z, y, physics) + precond.update_grad(
            {"stepsize": gamma}, physics, z - x
        )
        return gradient_descent(
            grad, x, step_size=stepsize_inter, max_iter=max_iter_inter, tol=tol_inter
        )
