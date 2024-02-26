import numpy as np


def calculate_ssim(img1, img2):
    # Check the shapes
    if img1.shape != img2.shape:
        raise ValueError("Görüntüler aynı boyutta olmalıdır.")

    # Compute Luminance
    K1 = 0.01
    K2 = 0.03
    L = 255  # Interval of pixels

    c1 = (K1 * L) ** 2
    c2 = (K2 * L) ** 2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = np.mean((img1 - mu1) ** 2)
    sigma2_sq = np.mean((img2 - mu2) ** 2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    # Compute the SSIM
    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_value = num / den

    return ssim_value
