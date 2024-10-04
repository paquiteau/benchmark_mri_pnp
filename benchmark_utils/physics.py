from deepinv.physics import LinearPhysics
from mrinufft.density.geometry_based import voronoi
import mrinufft
from tfkbnufft.kbnufft import KbNufftModule
from tfkbnufft.mri.dcomp_calc import calculate_density_compensator
from tfkbnufft import kbnufft_forward, kbnufft_adjoint
import numpy as np

NufftOperator = mrinufft.get_operator("gpunufft")

class Nufft(LinearPhysics):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    def __init__(
        self,
        img_size,
        samples_loc,
        density=None,
        real=False,
        n_coils=1,
        Smaps=None,
        **kwargs
    ):
        super(Nufft, self).__init__(**kwargs)

        self.real = real  # Whether to project the data on real images
        if density is not None:
            if density == 'voronoi':
                density = voronoi(samples_loc.reshape(-1, 2))
        if Smaps is not None:
            n_coils = len(Smaps)
        test_im = np.ones(img_size, dtype=np.complex64)
        test_nufft = NufftOperator(samples_loc.reshape(-1, 2), shape=img_size, density='pipe')
        test_im_recon = test_nufft.adj_op(
            test_nufft.op(test_im) 
        )
        ratio = np.mean(np.abs(test_im_recon))
        density = test_nufft.density / ratio
        self.nufft = NufftOperator(samples_loc.reshape(-1, 2), shape=img_size, density=density, n_coils=n_coils, squeeze_dims=False, smaps = Smaps)

    def A(self, x):
        return self.nufft.op(x)

    def A_adjoint(self, kspace):
        return self.nufft.adj_op(kspace)

