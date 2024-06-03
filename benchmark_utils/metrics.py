import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from deepinv.utils.metric import cal_psnr


def compute_ssim(imageA, imageB, mask):
    # Convert the images to grayscale
    # grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    # grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Ensure the images are of type np.uint8
    imageA, imageB = imageA.abs().squeeze(0).squeeze(0).numpy(), imageB.abs().squeeze(0).squeeze(0).numpy()
    # pdb.set_trace()
    if imageA.dtype != np.uint8:
        imageA = (255 * (imageA - imageA.min()) / (imageA.max() - imageA.min())).astype(np.uint8)
    if imageB.dtype != np.uint8:
        imageB = (255 * (imageB - imageB.min()) / (imageB.max() - imageB.min())).astype(np.uint8)

    # Compute SSIM between the two images
    score, diff = ssim(imageA * mask, imageB * mask, full=True)
    return score

def compute_psnr(target, x_hat):

    psnr = cal_psnr(target.abs(), x_hat.abs()).item()
    return psnr


