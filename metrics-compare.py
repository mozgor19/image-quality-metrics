import cv2
import matplotlib.pyplot as plt

from Filters.mean_filter import mean_filter
from Filters.median_filter import median_filter

from Noises.gaussian_noise import add_gaussian_noise
from Noises.salt_and_pepper import add_salt_and_pepper_noise
from Noises.spike_noise import add_spike_noise

from Metrics.ssim import calculate_ssim
from Metrics.psnr import calculate_psnr
from Metrics.snr import calculate_snr

image = cv2.imread("lena.png")
# image_same = cv2.imread("kaan.jpeg")

# Adding noise
gaussian_noised_image = add_gaussian_noise(image)
salted_image = add_salt_and_pepper_noise(image)
spike_noised_image = add_spike_noise(image)

kernel_size = 3

# Mean filter
mean_filtered_gaussian = mean_filter(gaussian_noised_image, kernel_size)
mean_filtered_gaussian_build = cv2.blur(
    gaussian_noised_image, (kernel_size, kernel_size)
)
mean_filtered_salt = mean_filter(salted_image, kernel_size)
mean_filtered_salt_build = cv2.blur(salted_image, (kernel_size, kernel_size))
mean_filtered_spike = mean_filter(spike_noised_image, kernel_size)
mean_filtered_spike_build = cv2.blur(spike_noised_image, (kernel_size, kernel_size))

# Median filter
median_filtered_gaussian = median_filter(gaussian_noised_image, kernel_size)
median_filtered_gaussian_build = cv2.medianBlur(gaussian_noised_image, kernel_size)
median_filtered_salt = median_filter(salted_image, kernel_size)
median_filtered_salt_build = cv2.medianBlur(salted_image, kernel_size)
median_filtered_spike = median_filter(spike_noised_image, kernel_size)
median_filtered_spike_build = cv2.medianBlur(spike_noised_image, kernel_size)


image_pairs = [
    (mean_filtered_gaussian, mean_filtered_gaussian_build),
    (mean_filtered_salt, mean_filtered_salt_build),
    (mean_filtered_spike, mean_filtered_spike_build),
    (median_filtered_gaussian, median_filtered_gaussian_build),
    (median_filtered_salt, median_filtered_salt_build),
    (median_filtered_spike, median_filtered_spike_build),
]

ssim_values = []
psnr_values = []
snr_values = []

for pair in image_pairs:
    img1 = pair[0]
    img2 = pair[1]

    ssim_index = calculate_ssim(img1, img2)
    psnr_value = calculate_psnr(img1, img2)
    snr_value = calculate_snr(img1, img2)

    ssim_values.append(ssim_index)
    psnr_values.append(psnr_value)
    snr_values.append(snr_value)


plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.plot(ssim_values)
plt.title("SSIM")
plt.xlabel("Görüntü Çifti")
plt.ylabel("SSIM Değeri")

plt.subplot(1, 3, 2)
plt.plot(psnr_values)
plt.title("PSNR")
plt.xlabel("Görüntü Çifti")
plt.ylabel("PSNR Değeri")

plt.subplot(1, 3, 3)
plt.plot(snr_values)
plt.title("SNR")
plt.xlabel("Görüntü Çifti")
plt.ylabel("SNR Değeri")

plt.tight_layout()
plt.show()
