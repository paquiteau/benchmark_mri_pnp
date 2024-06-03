from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:

    import fastmri
    import torch
    from benchmark_utils.physic import Nufft
    from benchmark_utils.utils import to_complex_tensor, match_image_stats, stand, Clip
    from deepinv.optim.data_fidelity import L2
    from deepinv.models import WaveletDictDenoiser
    from tqdm import tqdm

class Solver(BaseSolver):

    parameters = {
        'density': ['pipe'],
        'norm_init': [True],
        'norm_prior': [True, False]
    }

    def set_objective(self, kspace_data_hat, images, smaps, mask, samples_loc):

        self.kspace_data_hat = kspace_data_hat
        self.images = images
        self.smaps = smaps 
        self.samples_loc = samples_loc
        self.mask = mask

    def run(self, max_iter):

        physic_mcoil = Nufft((320,320), self.samples_loc, density=self.density, real=False, Smaps = self.smaps.squeeze(0).numpy())

        y = self.kspace_data_hat
        back = physic_mcoil.A_adjoint(y)
        if self.init_norm:
            back = match_image_stats(to_complex_tensor(self.images[0]), back) ### to better initialize the fista algorithm
            ### here we normalize the reverse image in the problem so it has the same statistics as the images we used
            ### to produce the y_hat, ie the kspace target

        if stepsize is None:
            stepsize = 1 / physic_mcoil.nufft.get_lipschitz_cst(max_iter = 20)

        data_fidelity = L2()
        a = 3  
        sigma = 0.00001
        # sigma = 0
        # stepsize = 0.1
        
        # Select a prior
        # wav = WaveletDictDenoiser(non_linearity="soft", level=5, list_wv=['db4', 'db8'], max_iter=10)
        wav = WaveletDictDenoiser(non_linearity="soft", level=6, list_wv=['db4', 'db8'], max_iter=15)

        device = 'cuda'
        denoiser = ComplexDenoiser(wav, norm).to(device)
            
        # Initialize algo variables
        x_cur = back.clone()
        w = back.clone()
        u = back.clone()
        
        # Lists to store the data fidelity and prior values
        data_fidelity_vals = []
        prior_vals = []
        L_x = [x_cur.clone()]

        # FISTA iteration
        with tqdm(total=max_iter) as pbar:
            for k in range(max_iter):
        
                tk = (k + a - 1) / a
                tk_ = (k + a) / a
        
                x_prev = x_cur.clone()
        
                x_cur = w - stepsize * data_fidelity.grad(w, y, physic_mcoil)
                x_cur = denoiser(x_cur, sigma * stepsize)
        
                w = (1 - 1 / tk) * x_cur + 1 / tk * u
        
                u = x_prev + tk * (x_cur - x_prev)
        
                crit = torch.linalg.norm(x_cur.flatten() - x_prev.flatten())

                # Compute and store data fidelity
                data_fidelity_val = data_fidelity(w, y, physic_mcoil)
                data_fidelity_vals.append(data_fidelity_val.item())

                # Compute and store prior value (for the denoiser)
                prior_val = sigma * stepsize * torch.sum(torch.abs(denoiser(x_cur, sigma * stepsize)))
                prior_vals.append(prior_val.item())
        
                pbar.set_description(f'Iteration {k}, criterion = {crit:.4f}')
                pbar.update(1)
                if k >= 0:
                    L_x.append(x_cur)
        
        x_hat = x_cur.clone()
        # return x_hat, data_fidelity_vals, prior_vals, L_x
        self.x_estimate = x_hat

    def get_result(self):
        return {'x_estimate':self.x_estimate}
    

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