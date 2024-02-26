import numpy as np
import cv2


def add_spike_noise(image, amount=0.01):
    """
    Function that adds Spike (Impulse) noise to an image.
    :param image: Image to which noise is to be added.
    :param amount: Amount of noise (default is 1%).
    :return: Image with added noise.
    """

    row, col, ch = image.shape
    noisy_image = np.copy(image)

    # Spike noise
    num_spike = np.ceil(amount * image.size)
    coords = [np.random.randint(0, i - 1, int(num_spike)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = np.random.choice([0, 255], ch)

    return noisy_image.astype(np.uint8)


# image = cv2.imread("image.jpg")
# noisy_image = add_spike_noise(image)

# cv2.imshow("Noisy Image", noisy_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
