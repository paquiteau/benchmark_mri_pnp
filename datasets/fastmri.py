"""Dataset for the fastmri multi-coil challenge."""

from benchopt import BaseDataset, safe_import_context
from pathlib import Path
import numpy as np
import os

with safe_import_context() as import_ctx:
    import torch
    from deepinv.physics.mri import ifft2c_new
    from deepinv.datasets.fastmri import FastMRISliceDataset
    from benchmark_utils.physics import Nufft
    from benchmark_utils.utils import complex_center_crop
    from skimage.morphology import (
        convex_hull_image,
        binary_closing,
        binary_dilation,
        disk,
    )
    from skimage.filters import threshold_otsu, gaussian

    from mrinufft.trajectories.trajectory2D import (
        initialize_2D_spiral,
        initialize_2D_radial,
    )

MAX_SAMPLES = 1


FASTMRI_PATH = os.environ.get(
    "FASTMRI_PATH",
    Path(__file__).parent.parent / "data" / "fastmri/multicoil_test_full",
)


class Dataset(BaseDataset):
    """Dataset for the fastmri multi-coil challenge."""

    name = "fastmri-mc"

    install_cmd = "conda"
    requirements = ["deepinv", "mri-nufft"]

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # parameters = {
    #     'n_samples, n_features': [
    #         (1000, 500),
    #         (5000, 200)]
    # }
    parameters = {
        "id": list(range(MAX_SAMPLES)),
        "contrast": ["FLAIR", "T1", "T2"],
        "sampling": ["spiral", "radial"],
        "AF": [4, 8, 16],
    }

    parameter_template = "{contrast}-{sampling}-{AF}"  # id is ommited to get average.

    def __init__(self, id, contrast, sampling, AF):
        self.id = id
        self.contrast = contrast
        self.sampling = sampling
        self.AF = AF

        self.dataloader = FastMRISliceDataset(
            root=FASTMRI_PATH,
            challenge="multicoil",
            test=False,
            load_metadata_from_cache=True,
            save_metadata_to_cache=True,
            sample_filter=lambda x: True,
            sample_rate=1.0,
        )
        rng = np.random.default_rng(42)
        if len(self.dataloader) > 0:
            random_ids = rng.choice(
                len(self.dataloader),
                size=min(MAX_SAMPLES, len(self.dataloader)),
                replace=False,
            )
        else:
            raise ValueError("Empty Dataset")
        self._fastmri_id = random_ids[id]

    def skip(self):
        if self.id >= min(MAX_SAMPLES, len(self.dataloader)):
            return True, "no such id"

    def get_data(self):
        target, full_kspace = self.dataloader[self._fastmri_id]
        # Get the VCC complex data
        # Get the smaps
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        if isinstance(full_kspace, np.ndarray):
            full_kspace = torch.from_numpy(full_kspace)
        full_image_channels = torch.view_as_complex(
            ifft2c_new(torch.view_as_real(full_kspace))
        )
        full_image_channels = complex_center_crop(full_image_channels, target.shape)
        full_image = virtual_coil_combination_2D(full_image_channels)
        if self.sampling == "spiral":
            samples_loc = initialize_2D_spiral(
                int(320 / self.AF), 320, nb_revolutions=1, in_out=True
            )
        if self.sampling == "radial":
            samples_loc = initialize_2D_radial(int(320 / self.AF), 320, in_out=True)
        self.smaps, mask = self.get_smaps(
            full_kspace, full_image, crop_size=target.shape
        )
        target *= mask
        # Initialize the physics model
        physics_sense = self.get_physics(target.shape, samples_loc, smaps=self.smaps)
        physics = self.get_physics(
            target.shape, samples_loc, n_coils=full_kspace.shape[0]
        )
        # Get the kspace data
        #
        noise_std = torch.std(target) * 1e-3
        kspace_data = physics.nufft.op(full_image_channels)

        kspace_data = kspace_data + torch.randn_like(kspace_data) * noise_std
        return dict(
            kspace_data=kspace_data,
            physics=physics_sense,
            target=target.cpu().numpy(),
            #    target=abs(full_image).cpu().numpy(),
            trajectory_name=self.sampling,
        )

    @staticmethod
    def get_smaps(full_kspace, full_image, crop_size):
        # Create a mask for the brain image
        magn = complex_center_crop(full_image, crop_size).abs().numpy()
        thresh = threshold_otsu(magn) * 0.8  # be conservative
        image_mask = magn > thresh
        image_mask = binary_dilation(image_mask, disk(20))
        image_mask = binary_closing(image_mask, disk(20))
        y_low = apply_hamming_filter(full_kspace, filter_size=(20, 20))

        images_low = complex_center_crop(ifft2c_new(y_low), crop_size)
        images_low = torch.view_as_complex(images_low)
        SOS = np.sum((np.abs(images_low.numpy()) ** 2), axis=0)
        Smaps_low = images_low / np.sqrt(SOS)
        Smaps_low *= image_mask
        return Smaps_low.detach().cpu().numpy(), image_mask

    @staticmethod
    def get_physics(image_shape, samples_loc, n_coils=1, smaps=None):
        physics = Nufft(
            image_shape, samples_loc, n_coils=n_coils, Smaps=smaps, density="pipe"
        )
        return physics


def virtual_coil_combination_2D(imgs, eps=1e-16):
    """
    Calculate the combination of all the coils using the virtual coil
    method for 2D images.

    Parameters
    ----------
    imgs: torch.Tensor
        The images reconstructed channel by channel [Nch, Nx, Ny]
    eps: float
        Small value to avoid division by zero

    Returns
    -------
    I: torch.Tensor
        The combination of all the channels in a complex valued [Nx, Ny]
    """
    # Ensure imgs is a complex tensor
    if not torch.is_complex(imgs):
        raise ValueError("Input tensor must be complex")

    # Compute the virtual coil
    nch, nx, ny = imgs.shape
    weights = torch.sum(torch.abs(imgs), dim=0, keepdim=True)
    weights = torch.clamp(weights, min=eps)

    phase_reference = torch.angle(torch.sum(imgs, dim=(1, 2), keepdim=True))
    reference = imgs / (weights * torch.exp(1j * phase_reference))
    virtual_coil = torch.sum(reference, dim=0)

    # Remove the background noise via low pass filtering
    hanning_2d = torch.outer(torch.hann_window(nx), torch.hann_window(ny))
    hanning_2d = torch.fft.fftshift(hanning_2d)[None, :, :]

    difference_original_vs_virtual = torch.conj(imgs) * virtual_coil
    difference_original_vs_virtual = torch.fft.ifft2(
        torch.fft.fft2(difference_original_vs_virtual, dim=(1, 2)) * hanning_2d,
        dim=(1, 2),
    )

    combined_imgs = torch.sum(
        imgs * torch.exp(1j * torch.angle(difference_original_vs_virtual)), dim=0
    )
    return combined_imgs


def apply_hamming_filter(kspace, filter_size):
    """
    Applies a Hamming window filter to the k-space data.

    Parameters:
    - kspace: A tensor of shape (1, 20, 640, 320, 2) representing the k-space data.
    - filter_size: A tuple (height, width) specifying the size of the low-frequency filter.

    Returns:
    - filtered_kspace: The Hamming window filtered k-space data with the same shape as input.
    """
    # Combine real and imaginary parts into a complex tensor
    kspace_complex = kspace.clone().detach()

    # Create the Hamming window filter
    hamming_window_1d_h = np.hamming(filter_size[0])
    hamming_window_1d_w = np.hamming(filter_size[1])
    hamming_window_2d = np.outer(hamming_window_1d_h, hamming_window_1d_w)

    # Pad the Hamming window to the size of the k-space data
    padded_hamming_window = np.zeros((kspace_complex.shape[1], kspace_complex.shape[2]))
    center_h = kspace_complex.shape[1] // 2
    center_w = kspace_complex.shape[2] // 2
    h_half = filter_size[0] // 2
    w_half = filter_size[1] // 2

    # Adjust the indices to correctly place the Hamming window at the center
    start_h = center_h - h_half
    end_h = start_h + filter_size[0]
    start_w = center_w - w_half
    end_w = start_w + filter_size[1]

    padded_hamming_window[start_h:end_h, start_w:end_w] = hamming_window_2d

    # Convert the Hamming window to a PyTorch tensor
    hamming_filter = torch.tensor(padded_hamming_window, dtype=torch.complex64).to(
        kspace_complex.device
    )

    # Apply the Hamming filter to the k-space data
    filtered_kspace_complex = kspace_complex * hamming_filter

    # Convert back to separate real and imaginary parts
    filtered_kspace = torch.view_as_real(filtered_kspace_complex)

    return filtered_kspace
