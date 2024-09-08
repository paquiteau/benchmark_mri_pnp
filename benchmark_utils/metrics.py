#!/usr/bin/env python3

from skimage.metrics import structural_similarity as ssim
from deepinv.utils import cal_psnr

import numpy as np


def compute_ssim(imageA, imageB):
    # Convert the images to grayscale
    # grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    # grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Ensure the images are of type np.uint8
    imageA = imageA.abs().squeeze(0).squeeze(0).numpy()
    imageB = imageB.abs().squeeze(0).squeeze(0).numpy()
    # pdb.set_trace()
    if imageA.dtype != np.uint8:
        imageA = (255 * (imageA - imageA.min()) / (imageA.max() - imageA.min())).astype(
            np.uint8
        )
    if imageB.dtype != np.uint8:
        imageB = (255 * (imageB - imageB.min()) / (imageB.max() - imageB.min())).astype(
            np.uint8
        )

    # Compute SSIM between the two images
    score, diff = ssim(imageA, imageB, full=True)
    return score


def compute_psnr(imageA, imageB, max_pixel=0.001):
    imageA = imageA.abs().squeeze(0).squeeze(0)
    imageB = imageB.abs().squeeze(0).squeeze(0)
    return cal_psnr(imageA, imageB, max_pixel=max_pixel)
