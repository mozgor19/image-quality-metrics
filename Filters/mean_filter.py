import numpy as np
import cv2


def mean_filter(image, kernel_size):
    height, width = image.shape[:2]
    kernel_half = kernel_size // 2
    filtered_image = np.zeros_like(image)
    padded_image = cv2.copyMakeBorder(
        image,
        kernel_half,
        kernel_half,
        kernel_half,
        kernel_half,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    for y in range(kernel_half, height + kernel_half):
        for x in range(kernel_half, width + kernel_half):
            roi = padded_image[
                y - kernel_half : y + kernel_half + 1,
                x - kernel_half : x + kernel_half + 1,
            ]
            filtered_image[y - kernel_half, x - kernel_half] = np.mean(roi)
    return filtered_image


# image = cv2.imread("kaan.jpeg", cv2.IMREAD_GRAYSCALE)
# kernel_size = 3
# filtered_image = mean_filter(image, kernel_size)
# cv2.imshow("Original Image", image)
# cv2.imshow("Mean Filtered Image", filtered_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
