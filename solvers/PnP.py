from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion, SufficientProgressCriterion
import pdb 
import numpy as np

with safe_import_context() as import_ctx:

    import fastmri
    import torch
    from benchmark_utils.physic import Nufft
    from benchmark_utils.utils import to_complex_tensor, match_image_stats, stand, Clip
    from deepinv.optim.data_fidelity import L2
    from deepinv.models import WaveletDictDenoiser
    from tqdm import tqdm
    from deepinv.models import DRUNet
    from deepinv.optim import optim_builder, PnP



'''
PnP
'''
sigmas = [1e-4,1e-2]

class Solver(BaseSolver):

    name = "PnP"
    sampling_strategy = "callback"
    stopping_criterion = SufficientProgressCriterion(patience=5)
    parameters = {
        'density': ['pipe'],
        'norm_init': [True],
        'norm_prior': [True],
        'prior': [None, 'PnP'],
        'iteration': ["HQS", "PGD", "FISTA"],
        # 'iteration': ["FISTA"],
        'sigma': sigmas,
        'max_iter': [int(1e3)],
    }

    def set_objective(self, kspace_data_hat, images, smaps, mask, kspace_mask, kspace_data, target):

        self.kspace_data_hat = kspace_data_hat
        self.images = images
        self.smaps = smaps 
        self.kspace_mask = kspace_mask
        self.mask = mask
        self.x_estimate = None
        self.it = 0
        self.physic_mcoil = Nufft((320,320), self.kspace_mask, density=self.density, real=False, Smaps = self.smaps.squeeze(0).numpy())

        self.y = self.kspace_data_hat
        self.back = self.physic_mcoil.A_adjoint(self.kspace_data_hat)
        if self.norm_init:
            self.back = match_image_stats(to_complex_tensor(self.images[0]), self.back) ### to better initialize the fista algorithm
            ### here we normalize the reverse image in the problem so it has the same statistics as the images we used
            ### to produce the y_hat, ie the kspace target

        self.stepsize = 1 / self.physic_mcoil.nufft.get_lipschitz_cst(max_iter = 20)

        self.data_fidelity = L2()
        

        # Initialize algo variables
        self.model = DRUNet(in_channels=1, out_channels=1, pretrained='download').to('cuda')
        self.model.eval()
        self.model = ComplexDenoiser(self.model, self.norm_prior)
        lamb, sigma_denoiser, stepsize, max_iter = get_DPIR_params(self.sigma, max_iter=self.max_iter, stepsize = self.stepsize)
        params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser, "lambda": lamb}
        early_stop = False  # Do not stop algorithm with convergence criteria
        if self.prior == 'PnP':
            self.prior = PnP(denoiser=self.model)
        self.algo = optim_builder(
        iteration = self.iteration,
        prior=self.prior,
        data_fidelity=self.data_fidelity,
        early_stop=early_stop,
        max_iter=max_iter,
        verbose=True,
        params_algo=params_algo,
        )

        self.x_cur = (
            self.algo.fixed_point.init_iterate_fn(self.y, self.physic_mcoil, F_fn= self.algo.fixed_point.iterator.F_fn)
        if  self.algo.fixed_point.init_iterate_fn
        else None
        )

    
    def run(self, callback):
        self.x_estimate = self.x_cur['est'][0]
        with torch.no_grad():
            while callback():   
                self.x_cur, _, check_iteration = self.algo.fixed_point.one_iteration(self.x_cur, self.it, self.y, self.physic_mcoil, compute_metrics = False, x_gt = None) 
                if check_iteration:
                    self.it += 1     
                self.x_estimate = self.x_cur['est'][0]
                if self.it >= 80:
                    break 

    def get_result(self):
        return {'x_estimate': self.x_estimate}
    
    def skip(self, **kwargs):

        if self.prior is None and self.sigma > np.min(sigmas):
            return True, "ok"
        else:
            return False, "redundant case"
    

class ComplexDenoiser(torch.nn.Module):
    def __init__(self, denoiser, norm):
        super().__init__()
        self.denoiser = denoiser
        self.norm = norm

    def forward(self, x, sigma):
        if self.norm:
            x_real, a_real, b_real = stand(x.real)
            x_imag, a_imag, b_imag = stand(x.imag)
        else:
            x_real, x_imag = x.real, x.imag
        noisy_batch = torch.cat((x_real, x_imag), 0)
        # noisy_batch, a, b = stand(noisy_batch)
        noisy_batch = noisy_batch.to('cuda')
        denoised_batch = self.denoiser(noisy_batch, sigma)
        # denoised_batch = denoised_batch * (b -a) + a
        if self.norm:
            denoised = (denoised_batch[0:1, ...] * (b_real - a_real) + a_real)+1j*(denoised_batch[1:2, ...] * (b_imag - a_imag) + a_imag)
        else:
            denoised = denoised_batch[0:1, ...]+1j*denoised_batch[1:2, ...] 
        return denoised.to('cpu')
    
def get_DPIR_params(noise_level_img, max_iter, stepsize):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    # s1 = 49.0 / 255.0
    s1 = 0.1
    s2 = noise_level_img
    # sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), max_iter).astype(
    #     np.float32
    # )
    xsi = 0.7
    sigma_denoiser = np.array([max(s1 * (xsi**i) , s2) for i in range(max_iter)]).astype(np.float32)
    # stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2 * 100
    stepsize = np.ones_like(sigma_denoiser) * stepsize
    # stepsize = (sigma_denoiser / max(0.01, noise_level_img)) * stepsize * 0.5
    lamb = 1 / 0.23
    # lamb = 1e8
    return lamb, list(sigma_denoiser), list(stepsize), max_iter
