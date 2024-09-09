from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion
import os
from pathlib import Path
import numpy as np

with safe_import_context() as import_ctx:
    import torch
    from deepinv.models import WaveletDictDenoiser
    from deepinv.optim.data_fidelity import L2
    from deepinv.optim import optim_builder, PnP, Prior
    from benchmark_utils.utils import stand
    from benchmark_utils.drunet import DRUNet

proj_dir = Path(__file__).parent.parent
DRUNET_PATH = os.environ.get("DRUNET_PATH", proj_dir / "drunet.tar")
DRUNET_DENOISE_PATH = os.environ.get(
    "DRUNET_DENOISE_PATH", proj_dir / "drunet_denoised.tar"
)


class Solver(BaseSolver):
    """Zero order solution"""

    name = "PNP"

    install_cmd = "conda"
    sampling_strategy = "callback"
    requirements = ["deepinv", "mrinufft[gpunufft]"]
    parameters = {
        "iteration": ["HQS", "PGD", "FISTA"],
        "prior": ["drunet", "drunet-denoised"],
    }
    max_iter = 10
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
        kwargs_optim["params_algo"] = get_DPIR_params(
            noise_level_img=0.1,
            n_iter=self.max_iter,
        )

        self.algo = optim_builder(
            iteration=self.iteration,
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
    from mrinufft.density import pipe

    density = pipe(physics.nufft.samples, shape=physics.nufft.shape, num_iterations=20)
    density = torch.from_numpy(density)
    est = physics.A_dagger(y)
    return {"est": (est, est.detach().clone())}


def get_DPIR_params(noise_level_img, s1=0.5, lamb=2, n_iter=10):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    s1 = 0.5
    s2 = noise_level_img
    sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), n_iter).astype(np.float32)
    stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2
    return {"lambda": lamb, "g_param": list(sigma_denoiser), "stepsize": list(stepsize)}
