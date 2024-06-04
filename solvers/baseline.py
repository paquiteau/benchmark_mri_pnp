from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SingleRunCriterion, SufficientProgressCriterion
import pdb 

with safe_import_context() as import_ctx:

    import fastmri
    import torch
    from benchmark_utils.physic import Nufft
    from benchmark_utils.utils import to_complex_tensor, match_image_stats, stand, Clip
    from deepinv.optim.data_fidelity import L2
    from deepinv.models import WaveletDictDenoiser
    from tqdm import tqdm

'''
Gradient Descent
'''
class Solver(BaseSolver):

    name = "baseline"
    sampling_strategy = "callback"
    parameters = {
        'density': ['pipe'],
        'norm_init': [True],
        'norm_prior': [True],
        'sigma': [1e-5, 1e-4]
    }
    stopping_criterion = SufficientProgressCriterion(patience=5)

    def set_objective(self, kspace_data_hat, images, smaps, mask, kspace_mask, **kwargs):

        self.kspace_data_hat = kspace_data_hat
        self.images = images
        self.smaps = smaps 
        self.kspace_mask = kspace_mask
        self.mask = mask
        self.x_estimate = None

        self.physic_mcoil = Nufft((320,320), self.kspace_mask, density=self.density, real=False, Smaps = self.smaps.squeeze(0).numpy())

        self.y = self.kspace_data_hat
        self.back = self.physic_mcoil.A_adjoint(self.kspace_data_hat)
        if self.norm_init:
            self.back = match_image_stats(to_complex_tensor(self.images[0]), self.back) ### to better initialize the fista algorithm
            ### here we normalize the reverse image in the problem so it has the same statistics as the images we used
            ### to produce the y_hat, ie the kspace target

        self.stepsize = 1 / self.physic_mcoil.nufft.get_lipschitz_cst(max_iter = 20)

        self.data_fidelity = L2()
        self.a = 3  
        # self.sigma = 0.001
        # sigma = 0
        # stepsize = 0.1
        
        # Select a prior
        # wav = WaveletDictDenoiser(non_linearity="soft", level=5, list_wv=['db4', 'db8'], max_iter=10)
        wav = WaveletDictDenoiser(non_linearity="soft", level=6, list_wv=['db4', 'db8'], max_iter=15)

        device = 'cuda'
        self.denoiser = ComplexDenoiser(wav, self.norm_prior).to(device)
        # Initialize algo variables
        self.x_cur = self.back.clone()
        self.w = self.back.clone()
        self.u = self.back.clone()
        print('end of set objective')
        # Lists to store the data fidelity and prior values
        # data_fidelity_vals = []
        # prior_vals = []
        # L_x = [x_cur.clone()]

        # FISTA iteration
        # while callback():
        self.k = 0
    


    def run(self, callback):
        self.x_prev = self.x_cur.clone()
        while callback():        
            tk = (self.k + self.a - 1) / self.a
           # tk_ = (self.k + self.a) / self.a
    
            self.x_prev = self.x_cur.clone()

            self.x_cur = self.w - self.stepsize * self.data_fidelity.grad(self.w, self.y, self.physic_mcoil)
            self.x_cur = self.denoiser(self.x_cur, self.sigma * self.stepsize)
    
            self.w = (1 - 1 / tk) * self.x_cur + 1 / tk * self.u
            self.u = self.x_prev + tk * (self.x_cur - self.x_prev)
            self.k += 1
                 
    def get_result(self):
        return {'x_estimate': self.x_cur}
    

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