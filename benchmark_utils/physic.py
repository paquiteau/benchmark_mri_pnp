import torch
from deepinv.physics import LinearPhysics
from mrinufft.density.geometry_based import voronoi

import mrinufft

# NufftOperator = mrinufft.get_operator("finufft")
NufftOperator = mrinufft.get_operator("gpunufft")

# NufftOperator = mrinufft.get_operator("cufinufft")


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
        Smaps=None,
        **kwargs
    ):
        super(Nufft, self).__init__(**kwargs)
        
        self.real = real  # Whether to project the data on real images
        if density is not None:
            if density == 'voronoi':
                density = voronoi(samples_loc.reshape(-1, 2))
        n_coils = 1
        if Smaps is not None:
            n_coils = len(Smaps)
        self.nufft = NufftOperator(samples_loc.reshape(-1, 2), shape=img_size, density=density, n_coils=n_coils, squeeze_dims=False, smaps = Smaps)

    def A(self, x):
        return self.nufft.op(x)

    def A_adjoint(self, kspace):
        return self.nufft.adj_op(kspace)

def corrupt(image, physics):

    if isinstance(image, torch.Tensor):
        # If image is already a tensor, add batch and channel dimensions
        image_torch = image.unsqueeze(0).unsqueeze(0)
    else:
    # If image is not a tensor, convert it to tensor and add batch and channel dimensions
        image_torch = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    
    x = image_torch.clone()
    
    # Generate the physics
    # physics = Nufft(image_torch[0, 0].shape, samples_loc, density=None)
    y = physics.A(x)
    back = physics.A_adjoint(y)
    return x, y, back

def corrupt_coils(images, physics):
    imgs = images.squeeze(0)  # because shape [1, n_img, 320, 320]
    n_img = imgs.shape[0]
    
    # Initialize tensors of zeros for x, y, and back
    X = torch.zeros(1, n_img, imgs.shape[1], imgs.shape[2], dtype=torch.complex64)
    Back = torch.zeros(1, n_img, imgs.shape[1], imgs.shape[2], dtype=torch.complex64)

    for i in range(n_img):
        x, y, back = corrupt(imgs[i], physics)
        # pdb.set_trace()
        if i == 0:
            Y = torch.zeros(1, n_img, y.shape[2], dtype=torch.complex64)

        # Update tensors with the values of x, y, and back
        X[0, i] = x
        Y[0, i] = y
        Back[0, i] = back

    return X, Y, Back


# def corrupt(image, samples_loc):

#     image_torch = torch.from_numpy(image).unsqueeze(0)
#     n_img = image_torch.shape[0]
#     L_x, L_y, L_back = [0] * n_img,[0] * n_img, [0] * n_img
#     for i in range(n_img):
#     # image_torch = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
#         x = image_torch[i].clone()
#         x = x/x.abs().max() #normalize the data, so it is in the range [0,1]
        
#         # Generate the physics
#         physics = Nufft(image_torch[i][0, 0].shape, samples_loc, density=None)
#         y = physics.A(x)
#         back = physics.A_adjoint(y)
#         L_x.append(x)
#         L_y.append(y)
#         L_back.append(back)
#     if n_img == 1:
#         return L_x[0], L_y[0], L_back[0]
#     else:
#         return L_x, L_y, L_back
    
