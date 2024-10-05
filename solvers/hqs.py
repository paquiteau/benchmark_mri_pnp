from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion
import os
from pathlib import Path
import numpy as np

with safe_import_context() as import_ctx:
    import torch
    from deepinv.models import WaveletDictDenoiser
    from deepinv.optim.data_fidelity import L2
    from deepinv.optim import optim_builder, PnP, Prior, BaseOptim
    from deepinv.optim.optim_iterators.hqs import fStepHQS, gStepHQS
    from deepinv.optim.optim_iterators import OptimIterator, fStep
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
        "iteration": ["classic"],
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
            df = L2()
            kwargs_optim["custom_init"] = get_custom_init_ppnp
            iterator = PreconditionedHQS(
                preconditioner=precond,
                prior=prior,
                data_fidelity=df,
                has_cost=False,
                F_fn=None,
                g_first=False,
            )
            self.algo = BaseOptim(
                iterator,
                has_cost=iterator.has_cost,
                data_fidelity=df,
                **kwargs_optim,
            )
        else:
            self.algo = optim_builder(
                iteration="HQS",
                prior=prior,
                data_fidelity=L2(),
                early_stop=False,
                custom_init=get_custom_init,
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
    return {"est": (est, est.detach().clone())}


def get_custom_init_ppnp(y, physics):

    est = physics.A_dagger(y)
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


class PreconditionedHQS(OptimIterator):
    """Implement the preconditioned PnP algorithm from Hong et al. 2024."""

    def __init__(self, preconditioner="dynamic", **kwargs):
        super().__init__(**kwargs)
        self.g_step = gStepHQS(**kwargs)
        self.f_step = fStepHQS(**kwargs)
        self.requires_prox_g = True
        if self.g_first:
            raise ValueError(
                "The preconditioned PnP algorithm should start with a step on f."
            )

        self.preconditioner = Precond(preconditioner)

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics):
        x_prev, x_prev_prev, grad_f_prev = X["est"]
        k = 0 if "it" not in X else X["it"]

        # TODO add the preconditioner step
        grad_f = cur_data_fidelity.grad(x_prev, y, physics)

        grad_f_precond = self.preconditioner.update_grad(
            cur_params, physics, grad_f, grad_f_prev, x_prev, x_prev_prev
        )
        z = x_prev - cur_params["stepsize"] * grad_f_precond
        x = self.g_step(z, cur_prior, cur_params)
        F = (
            self.F_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost
            else None
        )

        return {"est": (x, x_prev, grad_f), "cost": F, "it": k + 1}


class Precond:
    def __init__(self, name, theta1=0.2, theta2=2, delta=1 / 1.633):
        self.it = 0
        self.name = name
        self.theta1 = theta1
        self.theta2 = theta2
        self.delta = delta

    def get_alpha(self, s, m):
        alphas = np.linspace(0, 1, 1000)
        # line search of alpha

    def update_grad(self, cur_params, physics, grad, *args, **kwargs):
        if self.name == "static":
            grad = self._update_grad_static(cur_params, physics, grad, *args, **kwargs)
        elif self.name == "cheby":
            grad = self._update_grad_cheby(cur_params, physics, grad, *args, **kwargs)
        elif self.name == "dynamic":
            grad = self._update_grad_dynamic(cur_params, physics, grad, *args, **kwargs)
        return grad

    def _update_grad_static(self, cur_params, physics, grad_f, *args, **kwargs):
        """update the gradient with the static preconditioner"""

        alpha = cur_params["stepsize"]
        grad_f_preconditioned = physics.A_adjoint(physics.A(grad_f))

        grad_f_preconditioned *= -alpha
        grad_f_preconditioned += 2 * grad_f

        return grad_f_preconditioned

    def _update_grad_cheby(self, cur_params, physics, grad_f, *args, **kwargs):
        """update the gradient with the static cheby preconditioner"""

        alpha = cur_params["stepsize"]
        grad_f_preconditioned = physics.A_adjoint(physics.A(grad_f))
        grad_f_preconditioned *= -(10 / 3) * alpha
        grad_f_preconditioned += 4 * grad_f
        return grad_f_preconditioned

    def _update_grad_dynamic(self, cur_params, physics, grad_f, grad_f_prev, x, x_prev):
        """update the gradient with the dynamic preconditioner"""

        s = x - x_prev
        m = grad_f - grad_f_prev

        # precompute dot products
        sf = s.squeeze(0).squeeze(0).reshape(-1)
        mf = m.squeeze(0).squeeze(0).reshape(-1)

        ss = sf.dot(sf.conj()).real
        sm = sf.dot(mf.conj()).real
        mm = mf.dot(mf.conj()).real

        for a in np.linspace(0, 1, 1000):
            sv = a * ss + (1 - a) * sm
            vv = (a**2) * ss + ((1 - a) ** 2) * mm + (2 * a * (1 - a)) * sm
            if sv / ss >= self.theta1 and vv / sv <= self.theta2:
                break
        v = a * s + (1 - a) * m

        tau = ss / sv - torch.sqrt((ss / sv) ** 2 - ss / vv)

        tmp = sv - tau * vv
        grad_f_preconditioned = tau * grad_f

        if tmp >= self.delta * torch.sqrt(tmp * vv):
            u = sf - tau * vf
            u = u.dot(grad_f.squeeze(0).squeeze(0).reshape(-1)) * u
            u = u.reshape(grad_f.shape)

            grad_f_preconditioned += u / tmp
        return grad_f_preconditioned
