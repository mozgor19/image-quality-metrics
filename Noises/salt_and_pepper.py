import numpy as np
import cv2

def add_salt_and_pepper_noise(image, amount=0.04):
    """
    Function that adds Salt and Pepper noise to an image.
    :param image: Image to which noise is to be added.
    :param amount: Amount of noise (default is 4%).
    :return: Image with added noise.
    """

    row, col, ch = image.shape
    noisy_image = np.copy(image)

    # Salt
    num_salt = np.ceil(amount * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 255

    # Pepper
    num_pepper = np.ceil(amount * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 0

    return noisy_image.astype(np.uint8)

# image = cv2.imread("image.jpg")
# noisy_image = add_salt_and_pepper_noise(image)

# cv2.imshow("Noisy Image", noisy_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
