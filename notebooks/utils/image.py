import cv2
import numpy as np
from utils.enums import DatasetColorMode
from PIL import Image


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decodes an image file.
    Args:
        image_bytes: The bytes of the image.

    Returns: The array with image data with shape (height, width, channels),
        where channels can be 1 for grayscale or 3 for RGB.

    """
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_ANYCOLOR)
    if image.ndim == 2:
        # Reshape grayscale images of shape (height, width) to fit models format (height, width, channels)
        # Example: grayscale image of shape (420, 420) is reshaped to (420, 420, 1)
        image = image[:, :, np.newaxis]
    else:
        _, _, channels = image.shape
        if channels == 3:
            # BGR -> RGB (no Alpha)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def read_image_file(image_path: str) -> np.ndarray:
    """
    Reads and decodes an image file.
    Args:
        image_path: The path of the image.

    Returns: The array with image data with shape (height, width, channels),
        where channels can be 1 for grayscale or 3 for RGB.

    """
    with open(image_path, "rb") as file:
        return decode_image_bytes(file.read())


def write_image_file(image: np.ndarray, image_path: str):
    """
    Writes an image file.
    Args:
        image_bytes: The bytes of the image in RGB format with shape (height, width, channels).
        image_path: The path of the image.
    """
    _, _, channels = image.shape
    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(image_path, image)


def read_and_resize_image(image_path: str, image_size: tuple) -> np.array:
    """Reads an image and resize it, all images including grayscale are returned as RGB images."""
    image = read_image_file(image_path)
    image = cv2.resize(image, (image_size[1], image_size[0]))
    image = (
        cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if len(image.shape) == 2 or image.shape[2] == 1
        else image
    )
    return image


def read_and_resize_input_example(
    image_path: str, image_size: tuple, dataset_color_mode: str
) -> np.array:
    """Reads an image and resize it, adds or removes channels depending on required color_space.
    The output is a numpy array with shape (H, W, C) where C is 1 for GRAYSCALE and 3 for RGB.
    """
    image = read_image_file(image_path)
    image = cv2.resize(image, (image_size[1], image_size[0]))
    if dataset_color_mode == DatasetColorMode.COLOR:
        if len(image.shape) == 2 or image.shape[2] == 1:
            # OpenCV reads a monochrome image as a 2D array, but dataset color mode is COLOR,
            # because some images in a dataset were RGB.
            # The input example is converted to RGB to match the dataset color mode.
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif dataset_color_mode == DatasetColorMode.MONOCHROME:
        if len(image.shape) == 2:
            # OpenCV reads a monochrome image as a 2D array, dataset color mode is MONOCHROME.
            # A new axis is added to the input example, because the input example is expected to always have 3 channels.
            image = image[:, :, np.newaxis]
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # When files have 3 channels, but all the channels are the same, it is considered a MONOCHROME dataset.
            # The input example is converted to grayscale to match the dataset color mode.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]

    return image


def is_rgb_image(image_path: str) -> bool:
    """Checks if an image is RGB, images with 3 channels which are the same are considered grayscale."""
    # https://pillow.readthedocs.io/en/latest/handbook/concepts.html#modes
    # most images are grayscale
    with Image.open(image_path) as image:
        if image.mode == "L":
            return False

    # in the unlikely case where a file has 3 channels, but all the channels are the same, it is also considered to be grayscale
    image = read_image_file(image_path)
    return not (
        np.all(image[:, :, 0] == image[:, :, 1])
        and np.all(image[:, :, 0] == image[:, :, 2])
    )
