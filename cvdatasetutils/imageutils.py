from PIL import Image
import numpy as np


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def load_image(image_path):
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    return image, image_np