import torch
import numpy as np


def to_complex_tensor(tensor):
    # Check if the tensor has the expected shape
    if tensor.shape[-1] != 2:
        raise ValueError("The last dimension of the input tensor must have size 2 to form complex numbers.")
    
    # Split the tensor into real and imaginary parts
    real_part = tensor[..., 0]
    imaginary_part = tensor[..., 1]
    
    # Combine the real and imaginary parts into a complex tensor
    complex_tensor = torch.complex(real_part, imaginary_part)
    
    return complex_tensor

def stand(data, clip = None):

    if clip is None:
        clip = Clip()
        clip.quantile(data, 0.99)
    if np.iscomplexobj(data):
        data.real, a_real, b_real = __stand__(data.real, clip.real)
        data.imag, a_imag, b_imag =  __stand__(data.imag, clip.imag)
        return data, a_real, b_real, a_imag, b_imag
    elif data.shape[-1] == 2:
        data[...,0], data[...,1] = __stand__(data[...,0], clip.real), __stand__(data[...,1], clip.imag)
        ### TO DO a and b
    else:
        data, a, b = __stand__(data, clip.real)
    return data, a, b

def match_image_stats(img1, img2):

    mu1, std1 = (img1).mean(), (img1).std()
    mu2, std2 = (img2).mean(), (img2).std()

    img2 = (img2 - mu2) / std2
    img2 = img2 * std1 + mu1

    return img2
             
def __stand__(data, clip = None):

    if clip is not None:
        data = np.clip(data, a_min = clip[0], a_max = clip[1])
    a,b = data.min(), data.max()
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
    def __init__(self, real = None, imag = None):
        self.real = real 
        self.imag = imag
        # if self.real is None:
        #     self.quantile(data, q)

    def quantile(self, data, q):
        if check_type(data) == 'Array':
            data = torch.Tensor(data)
        if np.iscomplexobj(data):
            self.real = (torch.quantile(data.real, 1-q), torch.quantile(data.real, q))
            self.imag = (torch.quantile(data.imag, 1-q), torch.quantile(data.imag, q))
        elif data.shape[-1] == 2:
            self.real = (torch.quantile(data[...,0], 1-q), torch.quantile(data[...,0], q))
            self.imag = (torch.quantile(data[...,1], 1-q), torch.quantile(data[...,1], q))
        else:
            self.real = (torch.quantile(data, 1-q), torch.quantile(data, q))
