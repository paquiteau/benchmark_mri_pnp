#!/usr/bin/env python3
# This is a wrapper from https://github.com/sherbret/normalization_equivariant_nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class DRUNetEq(nn.Module):
    """Norm equivariant drunet."""

    def __init__(
        self,
        in_nc=1,
        out_nc=1,
        nc=[64, 128, 256, 512],
        nb=4,
        blind=False,
        mode="scale-equiv",
    ):
        super().__init__()

        bias = mode == "ordinary"
        self.blind = blind
        if not blind:
            in_nc += 1

        self.m_head = conv2d(
            in_nc, nc[0], 3, stride=1, padding=1, bias=bias, blind=blind, mode=mode
        )

        self.m_down = nn.ModuleList(
            [
                nn.Sequential(
                    *[ResBlock(nc[i], nc[i], bias=bias, mode=mode) for _ in range(nb)],
                    conv2d(
                        nc[i], nc[i + 1], 2, stride=2, padding=0, bias=bias, mode=mode
                    )
                )
                for i in range(len(nc) - 1)
            ]
        )

        self.m_body = nn.Sequential(
            *[ResBlock(nc[-1], nc[-1], bias=bias, mode=mode) for _ in range(nb)]
        )

        self.m_up = nn.ModuleList(
            [
                nn.Sequential(
                    upscale2(nc[i], nc[i - 1], bias=bias, mode=mode),
                    *[
                        ResBlock(nc[i - 1], nc[i - 1], bias=bias, mode=mode)
                        for _ in range(nb)
                    ]
                )
                for i in range(len(nc) - 1, 0, -1)
            ]
        )

        self.m_tail = conv2d(
            nc[0], out_nc, 3, stride=1, padding=1, bias=bias, mode=mode
        )

        self.res = nn.ModuleList([ResidualConnection(mode) for _ in range(len(nc))])

    def forward(self, x, sigma=None):
        # Size handling (h and w must divisible by d)
        # print('shape x', x.shape)
        # print('sigma', sigma)
        _, _, h, w = x.size()
        scale = len(self.m_down)
        d = 2**scale
        r1, r2 = h % d, w % d
        x = F.pad(
            x,
            pad=(0, d - r2 if r2 > 0 else 0, 0, d - r1 if r1 > 0 else 0),
            mode="constant",
            value=float(x.mean()),
        )

        if not self.blind:  # Concatenate noisemap as additional input
            if not isinstance(sigma, torch.Tensor):
                sigma = torch.Tensor([sigma])
            # sigma is tensor of shape (batch_size,) and needs to be broadcasted to (batch_size, 1, 1, 1)
            sigma = sigma.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            noisemap = sigma * torch.ones(
                x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype
            )
            x = torch.cat((x, noisemap), dim=1)

        layers = [self.m_head(x)]
        for i in range(scale):
            layers.append(self.m_down[i](layers[-1]))
        x = self.m_body(layers[-1])
        for i in range(scale):
            x = self.m_up[i](self.res[i](x, layers[-(1 + i)]))
        x = self.m_tail(self.res[-1](x, layers[0]))

        return x[..., :h, :w]


class AffineConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode="reflect",
        blind=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=False,
        )
        self.blind = blind

    def affine(self, w):
        """returns new kernels that encode affine combinations"""
        return (
            w.view(self.out_channels, -1).roll(1, 1).view(w.size())
            - w
            + 1 / w[0, ...].numel()
        )

    def forward(self, x):
        kernel = (
            self.affine(self.weight)
            if self.blind
            else torch.cat(
                (self.affine(self.weight[:, :-1, :, :]), self.weight[:, -1:, :, :]),
                dim=1,
            )
        )
        padding = tuple(
            elt for elt in reversed(self.padding) for _ in range(2)
        )  # used to translate padding arg used by Conv module to the ones used by F.pad
        padding_mode = (
            self.padding_mode if self.padding_mode != "zeros" else "constant"
        )  # used to translate padding_mode arg used by Conv module to the ones used by F.pad
        return F.conv2d(
            F.pad(x, padding, mode=padding_mode),
            kernel,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
        )


class AffineConvTranspose2d(nn.Module):
    """Affine ConvTranspose2d with kernel=2 and stride=2, implemented using PixelShuffle"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = AffineConv2d(in_channels, 4 * out_channels, 1)

    def forward(self, x):
        return F.pixel_shuffle(self.conv1x1(x), 2)


class SortPool(nn.Module):
    """Channel-wise sort pooling, C must be an even number"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # A trick with relu is used because the derivative for torch.aminmax is not yet implemented and torch.sort is slow.
        N, C, H, W = x.size()
        x1, x2 = torch.split(x.view(N, C // 2, 2, H, W), 1, dim=2)
        diff = F.relu(x1 - x2, inplace=True)
        return torch.cat((x1 - diff, x2 + diff), dim=2).view(N, C, H, W)


class ResidualConnection(nn.Module):
    """Residual connection"""

    def __init__(self, mode="ordinary"):
        super().__init__()

        self.mode = mode
        if mode == "norm-equiv":
            self.alpha = nn.Parameter(0.5 * torch.ones(1))

    def forward(self, x, y):
        if self.mode == "norm-equiv":
            return self.alpha * x + (1 - self.alpha) * y
        return x + y


def conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode="zeros",
    blind=True,
    mode="ordinary",
):
    if mode == "ordinary" or mode == "scale-equiv":
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias if mode == "ordinary" else False,
            padding_mode=padding_mode,
        )
    elif mode == "norm-equiv":
        return AffineConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode="reflect",
            blind=blind,
        )
    else:
        raise NotImplementedError(
            "Only ordinary, scale-equiv and norm-equiv modes are implemented"
        )


def upscale2(in_channels, out_channels, bias=True, mode="ordinary"):
    """Upscaling using convtranspose with kernel 2x2 and stride 2"""
    if mode == "ordinary" or mode == "scale-equiv":
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=bias if mode == "ordinary" else False,
        )
    elif mode == "norm-equiv":
        return AffineConvTranspose2d(in_channels, out_channels)
    else:
        raise NotImplementedError(
            "Only ordinary, scale-equiv and norm-equiv modes are implemented"
        )


def activation(mode="ordinary"):
    if mode == "ordinary" or mode == "scale-equiv":
        return nn.ReLU(inplace=True)
    elif mode == "norm-equiv":
        return SortPool()
    else:
        raise NotImplementedError(
            "Only ordinary, scale-equiv and norm-equiv modes are implemented"
        )


class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=False, mode="ordinary"):
        super().__init__()

        self.m_res = nn.Sequential(
            conv2d(
                in_channels, in_channels, 3, stride=1, padding=1, bias=bias, mode=mode
            ),
            activation(mode),
            conv2d(
                in_channels, out_channels, 3, stride=1, padding=1, bias=bias, mode=mode
            ),
        )

        self.sum = ResidualConnection(mode)

    def forward(self, x):
        return self.sum(x, self.m_res(x))
