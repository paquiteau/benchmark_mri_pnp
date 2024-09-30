from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion
import os
from pathlib import Path
import numpy as np

with safe_import_context() as import_ctx:
    import torch
    from deepinv.models import WaveletDictDenoiser
    from deepinv.optim.optim_iterators import OptimIterator, fStep
    from deepinv.optim.data_fidelity import L2
    from deepinv.optim import optim_builder, PnP, Prior, BaseOptim
    from benchmark_utils.utils import stand
    from benchmark_utils.drunet import DRUNet

proj_dir = Path(__file__).parent.parent
DRUNET_PATH = os.environ.get("DRUNET_PATH", proj_dir / "drunet.tar")
DRUNET_DENOISE_PATH = os.environ.get(
    "DRUNET_DENOISE_PATH", proj_dir / "drunet_denoised.tar"
)


class Solver(BaseSolver):
    """PnP Iteration using PGD or FISTA with DRUNet prior."""

    name = "PNP"

    install_cmd = "conda"
    sampling_strategy = "callback"
    requirements = ["deepinv", "mrinufft[gpunufft]"]
    parameters = {
        "iteration": ["PGD", "FISTA", "ppnp-static", "ppnp-cheby", "ppnp-dynamic"],
        "prior": ["drunet", "drunet-denoised"],
    }
    max_iter = 20
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
        self.kspace_data = kspace_data
        self.physics = physics
        kwargs_optim = dict()
        denoiser = load_drunet(
            DRUNET_DENOISE_PATH if "denoised" in self.prior else DRUNET_PATH
        )
        cpx_denoiser = Denoiser(denoiser)
        prior = PnP(cpx_denoiser)
        kwargs_optim["params_algo"] = {
            "lambda": 2,  # f + lambda * g(x, g_params)
            "g_param": 0.1,
            "stepsize": 1 / self.physics.nufft.get_lipschitz_cst(10),
        }
        kwargs_optim["early_stop"] = False
        kwargs_optim["verbose"] = False
        kwargs_optim["custom_init"] = get_custom_init
        kwargs_optim["max_iter"] = self.max_iter

        if "pnpp" in self.iteration:
            _, precond = self.iteration.split("-")
            df = L2()
            iterator = PreconditionedPnP(
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
                iteration=self.iteration,
                prior=prior,
                data_fidelity=L2(),
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


def get_DPIR_params(s1=0.5, s2=0.1, lamb=2, n_iter=10):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    s1 = 0.5
    s2 = noise_level_img
    sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), n_iter).astype(np.float32)
    stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2
    return {"lambda": lamb, "g_param": list(sigma_denoiser), "stepsize": list(stepsize)}


class PreconditionedPnP(OptimIterator):
    """Implement the preconditioned PnP algorithm from Hong et al. 2024."""

    def __init__(self, preconditioner="dynamic", **kwargs):
        super().__init__(**kwargs)
        self.g_step = gStepPGD(**kwargs)
        self.f_step = fStepPGD(**kwargs)
        self.requires_prox_g = True
        if self.g_first:
            raise ValueError(
                "The preconditioned PnP algorithm should start with a step on f."
            )

        self.preconditionner = Precond(preconditioner)

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics):
        x_prev, grad_f_prev, precond_prev = X["est"]
        k = 0 if "it" not in X else X["it"]

        # TODO add the preconditioner step
        grad_f = self.f_step(
            z_prev, cur_data_fidelity, cur_params, y, physics, self.preconditioner
        )

        grad_f_precond = self.preconditioner.update_grad(
            cur_params, physics, grad_f, grad_f_prev, x, x_prev
        )
        z = x - cur_params["stepsize"] * grad_f_precond

        x = self.g_step(z, cur_prior, cur_params)
        z = x + alpha * (x - x_prev)
        F = (
            self.F_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost
            else None
        )

        return {"est": (x, grad_f, precond), "cost": F, "it": k + 1}


class Precond:
    def __init__(self, name, theta1=0.2, theta2=2, delta=1 / 1.633):
        self.it = 0
        self.name = name

    def get_alpha(self, s, m):
        alphas = np.linspace(0, 1, 1000)
        # line search of alpha

    def update_grad(self, cur_params, physics, grad, *args, **kwargs):
        if self.name == "static":
            grad = self._update_grad_static(physics, grad)

    def _update_grad_static(self, cur_params, physics, grad_f, *args, **kwargs):
        """update the gradient with the static preconditioner"""

        alpha = cur_params["stepsize"]
        grad_f_preconditioned = physics.A_adjoint(physics.A(grad_f))

        grad_f_preconditioned *= -alpha
        grad_f_preconditioned += 2 * grad_f

        return grad_f_preconditioned

    def _update_grad_cheby(self, cur_params, physics, grad_f):
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

        ss = sf.dot(sf)
        sm = sf.dot(mf)
        mm = mf.dot(mf)

        for a in np.linspace(0, 1, 1000):
            sv = a * ss + (1 - a) * sm
            vv = (a**2) * ss + ((1 - a) ** 2) * mm + (2 * a * (1 - a)) * sm
            if sv / ss >= self.theta1 and v.dot(v) / sv <= self.theta2:
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
