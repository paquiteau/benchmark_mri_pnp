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
    stopping_criterion = SufficientProgressCriterion(patience=5)
    parameters = {
        'density': ['pipe'],
        'norm_init': [True],
        'norm_prior': [True],
    }

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
        
        # Initialize algo variables
        self.x_cur = self.back.clone()
    
    def run(self, callback):
        while callback():         
            self.x_cur = self.x_cur - self.stepsize * self.data_fidelity.grad(self.x_cur, self.y, self.physic_mcoil)
              
    def get_result(self):
        return {'x_estimate': self.x_cur}