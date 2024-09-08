"""utility functions for data processing and visualization"""

import numpy as np
import torch


def stand(data, clip=None):

    if clip is None:
        clip = Clip()
        clip.quantile(data, 0.99)
    if np.iscomplexobj(data):
        data.real, a_real, b_real = _stand(data.real, clip.real)
        data.imag, a_imag, b_imag = _stand(data.imag, clip.imag)
        return data, a_real, b_real, a_imag, b_imag
    elif data.shape[-1] == 2:
        data[..., 0], data[..., 1] = _stand(data[..., 0], clip.real), _stand(
            data[..., 1], clip.imag
        )
        ### TO DO a and b
    else:
        data, a, b = _stand(data, clip.real)
    return data, a, b


def match_image_stats(img1, img2):

    mu1, std1 = (img1).mean(), (img1).std()
    mu2, std2 = (img2).mean(), (img2).std()

    img2 = (img2 - mu2) / std2
    img2 = img2 * std1 + mu1

    return img2


def _stand(data, clip=None):

    if clip is not None:
        data = np.clip(data, a_min=clip[0], a_max=clip[1])
    a, b = data.min(), data.max()
    return (data - a) / (b - a), a, b


def check_type(obj):
    if isinstance(obj, torch.Tensor):
        return "Tensor"
    elif isinstance(obj, np.ndarray):
        return "Array"
    else:
        return "Unknown Type"


class Clip:

    # def __init__(self, data, real = None, imag = None, q = 0.99):
    def __init__(self, real=None, imag=None):
        self.real = real
        self.imag = imag
        # if self.real is None:
        #     self.quantile(data, q)

    def quantile(self, data, q):
        if check_type(data) == "Array":
            data = torch.Tensor(data)
        if np.iscomplexobj(data):
            self.real = (torch.quantile(data.real, 1 - q), torch.quantile(data.real, q))
            self.imag = (torch.quantile(data.imag, 1 - q), torch.quantile(data.imag, q))
        elif data.shape[-1] == 2:
            self.real = (
                torch.quantile(data[..., 0], 1 - q),
                torch.quantile(data[..., 0], q),
            )
            self.imag = (
                torch.quantile(data[..., 1], 1 - q),
                torch.quantile(data[..., 1], q),
            )
        else:
            self.real = (torch.quantile(data, 1 - q), torch.quantile(data, q))


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def complex_center_crop(data: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if torch.is_complex(data):
        data_ = torch.view_as_real(data)
        data_ = complex_center_crop(data_, shape)
        return torch.view_as_complex(data_)

    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]
