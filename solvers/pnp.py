from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion
import os
from pathlib import Path
import numpy as np
from benchmark_utils.precond import Precond

with safe_import_context() as import_ctx:
    import torch
    from deepinv.models import WaveletDictDenoiser
    from deepinv.optim.optim_iterators import OptimIterator, fStep
    from deepinv.optim.optim_iterators.pgd import gStepPGD, fStepPGD
    from deepinv.optim.data_fidelity import L2
    from deepinv.optim import optim_builder, PnP, Prior, BaseOptim
    from benchmark_utils.utils import stand
    from benchmark_utils.drunet import DRUNet
    from benchmark_utils.druneteq import DRUNeteq

weight_dir = Path(__file__).parent.parent / "model_weights"
DRUNET_PATH = os.environ.get("DRUNET_PATH", weight_dir / "drunet_noisy.tar")
DRUNET_DENOISE_PATH = os.environ.get(
    "DRUNET_DENOISE_PATH", weight_dir / "drunet_clean.tar"
)
DRUNET_EQ_PATH = os.environ.get("DRUNET_EQ_PATH", weight_dir / "drunet_eq.tar")


class Solver(BaseSolver):
    """PnP Iteration using PGD or FISTA with DRUNet prior."""

    name = "PNP"

    install_cmd = "conda"
    sampling_strategy = "callback"
    requirements = ["deepinv", "mrinufft[gpunufft]"]
    parameters = {
        "iteration": ["PGD", "FISTA", "ppnp-static", "ppnp-cheby", "ppnp-dynamic"],
        "prior": ["drunet", "drunet-denoised", "drunet-eq"],
        "max_iter": [50],
    }
    stopping_criterion = SufficientProgressCriterion(patience=100)

    def skip(self, *args, **kwargs):
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
        x_init,
    ):
        self.kspace_data = kspace_data
        self.physics = physics
        kwargs_optim = dict()

        if self.prior == "drunet":
            denoiser = load_drunet(DRUNET_PATH)
        elif self.prior == "drunet-denoised":
            denoiser = load_drunet(DRUNET_DENOISE_PATH)
        elif self.prior == "drunet-eq":
            denoiser = load_drunet_eq(DRUNET_EQ_PATH)
        cpx_denoiser = Denoiser(denoiser)
        prior = PnP(cpx_denoiser)
        kwargs_optim["params_algo"] = {
            "lambda": 2,  # f + lambda * g(x, g_params)
            "g_param": 0.1,
            "stepsize": 1 / self.physics.nufft.get_lipschitz_cst(10),
        }
        kwargs_optim["early_stop"] = False
        kwargs_optim["verbose"] = False
        kwargs_optim["custom_init"] = lambda y, p: {
            "est": (x_init, x_init.detach().clone())
        }
        kwargs_optim["max_iter"] = self.max_iter

        if "ppnp" in self.iteration:
            _, precond = self.iteration.split("-")
            df = L2()
            kwargs_optim["custom_init"] = lambda y, p: {
                "est": (x_init, x_init.detach().clone(), torch.zeros_like(x_init))
            }
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


class DenoiserEq(Denoiser):
    """Equivariant Denoiser to wrap DRUNET for Complex data."""

    def forward(self, x, sigma, factor=1e4):
        x = torch.permute(torch.view_as_real(x.squeeze(0)), (0, 3, 1, 2)).to("cuda")
        x = x * factor
        x_ = torch.permute(self.denoiser(x).to("cpu"), (0, 2, 3, 1))
        x_ = x_ * 1 / factor
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


def load_drunet_eq(path_weights):

    model = DRUNetEq(in_nc=2, out_nc=2)
    file_name = "ckp_3037_eq.pth.tar"
    checkpoint = torch.load(file_name, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)

    sigma = 0.5
    fact = 1e4


# def get_custom_init(y, physics):

#     est = physics.A_dagger(y)
#     # est = torch.zeros_like(est)
#     return {"est": (est, est.detach().clone())}


# def get_custom_init_ppnp(y, physics):

#     est = physics.A_dagger(y)
#     # est = torch.zeros_like(est)
#     # return {
#     #     "est": (torch.zeros_like(est), torch.zeros_like(est), torch.zeros_like(est))
#     # }
#     return {"est": (est, est.detach().clone(), torch.zeros_like(est))}


def get_DPIR_params(s1=0.5, s2=0.1, lamb=2, n_iter=10):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    s1 = 0.5
    sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), n_iter).astype(np.float32)
    stepsize = (sigma_denoiser / max(0.01, s2)) ** 2
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
