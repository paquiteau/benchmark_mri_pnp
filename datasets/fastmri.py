
from benchopt import BaseDataset

from benchopt import safe_import_context
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np

with safe_import_context() as import_ctx:
    import torch
    from fastmri.data.subsample import MaskFunc
    from fastmri.data import SliceDataset
    import fastmri
    import pathlib
    from fastmri.data import subsample
    from fastmri.data import transforms, mri_data
    from fastmri.data.transforms import to_tensor, complex_center_crop
    import fastmri.data.transforms as T
    from benchmark_utils.physic import Nufft
    from benchmark_utils.kspace_sampling import get_samples
    from benchmark_utils.utils import to_complex_tensor
    # from physic import corrupt_coils 
    import cv2
    # from data2 import ClassicDataTransform

path = '/neurospin/optimed/BenjaminLapostolle/fast-mri_smal/'

class Dataset(BaseDataset):
    name = "fastmri"
    parameters = {
        "idx": [0],
        "Nc": [50],
        "Ns": [1000],
        "traj": [3],
        "density": ["pipe"]
                  }
    requirements = ['fastmri', 'torch', 'cv2', 'mrinufft', 'deepinv']

    def get_data(self):

        traj = ['sparkling', 'cartesian', 'radial', 'spiral', 'cones', 'sinusoide', 'propeller', 'rings', 'rosette', 'polar_lissajous', 'lissajous', 'waves']
        samples_loc = get_samples(traj[self.traj], Nc = self.Nc, Ns = self.Ns)
        physics = Nufft((320,320), samples_loc, density=self.density, real=False, Smaps = None)

        data_transform = ClassicDataTransform(which_challenge="multicoil", Smaps = True, physics = physics)
        dataset = mri_data.SliceDataset(
            root=pathlib.Path(path),
            transform=data_transform,
            challenge='multicoil'
        )

        dataloader = torch.utils.data.DataLoader(dataset)
        i = -1
        for batch in dataloader:
            if self.idx != i:
                i += 1
                continue    
            target, images, y, y_hat, Smaps, mask = batch
            y_hat = torch.cat(y_hat, dim = 0).unsqueeze(0)

            return dict(kspace_data=y, kspace_data_hat=y_hat, target=target, images=images, smaps=Smaps, mask=mask, kspace_mask=samples_loc)
  
class ClassicDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        Smaps: bool = False,
        physics = None,
        use_seed: bool = True,
        use_abs: bool = False,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.Smaps = Smaps 
        self.physics = physics 

        

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """
        y = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # inverse Fourier transform to get zero filled solution
        images_nostand = fastmri.ifft2c(y)
        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if images_nostand.shape[-2] < crop_size[1]:
            crop_size = (images_nostand.shape[-2], images_nostand.shape[-2])

        if target is not None:
            target_torch = to_tensor(target)
            target_torch = T.center_crop(target_torch, crop_size)
        else:
            target_torch = torch.Tensor([0])

        images_nostand = T.complex_center_crop(images_nostand, crop_size)
        ### normalization 

        # clip = Clip()
        # clip.quantile(images_nostand, 0.99)
        # images = stand(images_nostand.clone(), clip)
        images = images_nostand.clone()

        if self.physics is None:
            return target_torch, images, y
        
        y_hat = []
        for n_coil in range(y.shape[0]):
            y_hat.append(self.physics.A(T.tensor_to_complex_np(images[n_coil])).squeeze(0).squeeze(0))

        if not(self.Smaps):
            return target_torch, images, y, y_hat

        mask = self.__compute_mask__(images_nostand)
        Smaps = self.compute_Smaps(images, y_hat, mask)
        return target_torch, images, y, y_hat, Smaps, mask
    
    def compute_Smaps(self, images, y_hat, mask):
        
        images_hat = T.tensor_to_complex_np(images)
        # images_hat = np.zeros_like(images[:, :, :, 0], dtype=np.complex64)

        # for n in range(images.shape[0]):
        #     images_hat[n] = self.physics.A_adjoint(y_hat[n])[0,0]
        SOS = np.sum((np.abs(images_hat)**2), axis = 0)
        Smaps = images_hat / np.sqrt(SOS)
        return Smaps * mask
    
    def __compute_mask__(self, images):

        '''
        Args 
        images is the complex images obtained from the full FFT of the kspace y. We take the complete kspace and not the undersampled kspace_hat 
        because we assume we can could access this data experimentally 
        '''
        image = fastmri.complex_abs(images)
        # apply Root-Sum-of-Squares if multicoil data
        image = fastmri.rss(image).numpy()

        ### could use Otsu instead of approxiamte quantile 
        q = np.quantile(image, 0.6)
        image_bin = image.copy()
        image_bin[image > q] = 1
        image_bin[image <= q] = 0
        contours, _ = cv2.findContours(image_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        brain_mask = cv2.drawContours(np.zeros_like(image_bin.astype(np.uint8)), contours, -1, (255), thickness=cv2.FILLED)
        brain_mask[brain_mask == 255] = 1
        return brain_mask


    