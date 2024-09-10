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
    """FISTA Wavelet."""

    name = "FISTA-wavelet"

    install_cmd = "pip"
    sampling_strategy = "callback"
    requirements = ["deepinv", "mrinufft[gpunufft]"]
    parameters = {"sigma": [1e-6]}
    max_iter = 50
    a = 3  # From Chambolle's FISTA
    stopping_criterion = SufficientProgressCriterion(patience=30)

    def get_next(self, stop_val):
        return stop_val + 1

    def set_objective(self, kspace_data, physics, trajectory_name):
        self.kspace_data = kspace_data
        self.physics = physics
        wavelet = WaveletDictDenoiser(non_linearity="soft", level=6, list_wv=["sym8"])
        self.denoiser = ComplexDenoiser(wavelet, True).to("cuda")
        self.data_fidelity = L2()

    def run(self, callback):
        with torch.no_grad():
            # x_cur = get_custom_init(self.kspace_data, self.physics)
            x_cur = self.physics.A_dagger(self.kspace_data)
            self.stepsize = self.physics.nufft.get_lipschitz_cst()
            self.x_estimate = x_cur.clone()
            z = self.x_estimate.detach().clone()
            self.itr = 0
            self.cost = None
            while callback():
                # Fista iteration with wavelet prior
                alpha = (self.itr + self.a - 1) / (self.itr + self.a)
                x_cur = z - self.stepsize * self.data_fidelity.grad(
                    z, self.kspace_data, self.physics
                )
                x_cur = self.denoiser(x_cur, self.sigma * self.stepsize)
                z = x_cur + alpha * (x_cur - self.x_estimate)

                self.x_estimate = x_cur.clone()
                self.itr += 1

    def get_result(self):
        """Get values to pass to objective."""
        return {
            "x_estimate": self.x_estimate.cpu().numpy(),
            "cost": self.cost,
        }

    def _get_estimate(self, x_cur):
        x_est = x_cur["est"]
        if isinstance(x_est, tuple):
            x_est = x_est[1]
        return x_est, x_cur["cost"]


class ComplexDenoiser(torch.nn.Module):
    """Apply a denoiser to complex data, by splitting the real and imaginary parts."""

    def __init__(self, denoiser, norm: bool):
        super().__init__()
        self.denoiser = denoiser
        self.norm = norm

    def forward(self, x, *args, **kwargs):
        if self.norm:
            x_real, a_real, b_real = stand(x.real)
            x_imag, a_imag, b_imag = stand(x.imag)
        else:
            x_real, x_imag = x.real, x.imag
        noisy_batch = torch.cat((x_real, x_imag), 0)
        # noisy_batch, a, b = stand(noisy_batch)
        noisy_batch = noisy_batch.to("cuda")
        denoised_batch = self.denoiser(noisy_batch, *args, **kwargs)
        # denoised_batch = denoised_batch * (b -a) + a
        if self.norm:
            denoised = (denoised_batch[0:1, ...] * (b_real - a_real) + a_real) + 1j * (
                denoised_batch[1:2, ...] * (b_imag - a_imag) + a_imag
            )
        else:
            denoised = denoised_batch[0:1, ...] + 1j * denoised_batch[1:2, ...]
        return denoised.to("cpu")
