import numpy as np
import cv2


def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Function that adds Gaussian noise to an image.
    :param image: Image to which noise is to be added.
    :param mean: Mean value for Gaussian noise.
    :param sigma: Standard deviation for Gaussian noise.
    :return: Image with added noise.
    """

    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = np.clip(image + gauss, 0, 255)
    return noisy_image.astype(np.uint8)


# image = cv2.imread("image.jpg")
# noisy_image = add_gaussian_noise(image)
# cv2.imshow("Noisy Image", noisy_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
