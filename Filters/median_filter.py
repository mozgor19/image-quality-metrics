import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def median_filter(image, kernel_size):
    padded_image = np.pad(image, kernel_size // 2, mode="edge")
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded_image[i : i + kernel_size, j : j + kernel_size]
            median_value = np.median(neighborhood)
            filtered_image[i, j] = median_value
    return filtered_image


# # Load the image
# image = np.array(Image.open("kaan.jpeg").convert("L"))  # Convert to grayscale

# # Apply median filter with kernel size 3
# filtered_image = median_filter(image, kernel_size=3)

# # Display the original and filtered image
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap="gray")
# plt.title("Original Image")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(filtered_image, cmap="gray")
# plt.title("Filtered Image")
# plt.axis("off")

# plt.show()
