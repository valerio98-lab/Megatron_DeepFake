"""Module containing the definition of the transformations that are applied to the images"""

import random
from typing import Callable, Union, Tuple, List
import numpy as np
import cv2


def identity(image) -> np.ndarray:
    return image


def rotate(image: np.ndarray, angle: float = 45) -> np.ndarray:
    """
    Rotates an image by a given angle around its center.

    Parameters:
    - image (np.ndarray): The input image to be rotated.
    - angle (float): The angle of rotation in degrees (default 45).

    Returns:
    - np.ndarray: The rotated image.
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (width, height))
    return rotated


def horizontal_flip(image: np.ndarray) -> np.ndarray:
    """
    Performs a horizontal flip of the image.

    Parameters:
    - image (np.ndarray): The input image to flip.

    Returns:
    - np.ndarray: The horizontally flipped image.
    """
    return cv2.flip(image, 1)


def vertical_flip(image: np.ndarray) -> np.ndarray:
    """
    Performs a vertical flip of the image.

    Parameters:
    - image (np.ndarray): The input image to flip.

    Returns:
    - np.ndarray: The vertically flipped image.
    """
    return cv2.flip(image, 0)


def resize(image: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Resizes an image to a specified size.

    Parameters:
    - image (np.ndarray): The input image to be resized.
    - size (Tuple[int, int]): The target size as a tuple (width, height), default is (256, 256).

    Returns:
    - np.ndarray: The resized image.
    """
    return cv2.resize(image, size)


def crop(
    image: np.ndarray,
    start_x: int = 50,
    start_y: int = 50,
    width: int = 200,
    height: int = 200,
) -> np.ndarray:
    """
    Crops an image to a specified rectangle.

    Parameters:
    - image (np.ndarray): The input image to be cropped.
    - start_x (int): The x-coordinate of the top-left corner of the crop rectangle.
    - start_y (int): The y-coordinate of the top-left corner of the crop rectangle.
    - width (int): The width of the crop rectangle.
    - height (int): The height of the crop rectangle.

    Returns:
    - np.ndarray: The cropped image.
    """
    return image[start_y : start_y + height, start_x : start_x + width]


def adjust_brightness(image: np.ndarray, value: int = 50) -> np.ndarray:
    """
    Adjusts the brightness of an image.

    Parameters:
    - image (np.ndarray): The input image whose brightness is to be adjusted.
    - value (int): The value to adjust the brightness, higher values make the image brighter.

    Returns:
    - np.ndarray: The brightness adjusted image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Safely adjust the V channel to increase brightness
    v = np.clip(v + value, 0, 255)  # Ensures that the values are within the valid range

    # Reconstruct the HSV image and convert back to BGR color space
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def perspective_transform(image: np.ndarray) -> np.ndarray:
    """
    Applies a perspective transformation to an image to simulate a 3D effect.

    Parameters:
    - image (np.ndarray): The input image to transform.

    Returns:
    - np.ndarray: The perspective transformed image.
    """
    rows, cols = image.shape[:2]
    src_points = np.array(
        [
            [0, 0],
            [cols - 1, 0],
            [0, rows - 1],
            [cols - 1, rows - 1],
        ],
        dtype=np.float32,
    )
    dst_points = np.array(
        [
            [0, 0],
            [int(0.9 * cols), 0],
            [int(0.1 * cols), rows - 1],
            [cols - 1, rows - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed = cv2.warpPerspective(image, matrix, (cols, rows))
    return transformed


def random_erasing(
    image: np.ndarray,
    probability: float = 0.5,
    sl: float = 0.02,
    sh: float = 0.4,
    r1: float = 0.3,
) -> np.ndarray:
    """
    Randomly erases a rectangle portion of the image.

    Parameters:
    - image (np.ndarray): The input image to be erased.
    - probability (float): The probability of erasing.
    - sl (float): Minimum fraction of erased area.
    - sh (float): Maximum fraction of erased area.
    - r1 (float): Aspect ratio of the erased area.

    Returns:
    - np.ndarray: The image with a portion randomly erased.
    """
    if np.random.rand() > probability:
        return image
    image = image.copy()
    area = image.shape[0] * image.shape[1]
    target_area = np.random.uniform(sl, sh) * area
    aspect_ratio = np.random.uniform(r1, 1 / r1)
    h = int(np.sqrt(target_area * aspect_ratio))
    w = int(np.sqrt(target_area / aspect_ratio))
    if w < image.shape[1] and h < image.shape[0]:
        x1 = np.random.randint(0, image.shape[1] - w)
        y1 = np.random.randint(0, image.shape[0] - h)
        image[y1 : y1 + h, x1 : x1 + w, :] = 0
    return image


def add_gaussian_noise(image: np.ndarray) -> np.ndarray:
    """
    Adds Gaussian noise to an image.

    Parameters:
    - image (np.ndarray): The input image to add noise.

    Returns:
    - np.ndarray: The noisy image.
    """
    row, col, ch = image.shape
    mean = 0
    var = 0.01
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss * 255
    return noisy.astype("uint8")


def adjust_contrast(image: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Adjusts the contrast of an image.

    Parameters:
    - image (np.ndarray): The input image whose contrast is to be adjusted.
    - factor (float): The factor by which the contrast will be adjusted.

    Returns:
    - np.ndarray: The contrast adjusted image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = int(np.mean(gray))
    contrast = cv2.addWeighted(image, factor, image, 0, 128 - mean)
    return contrast


ParameterGenerator = Callable[[], dict[str, Union[int, float]]]
TransformationFunction = Callable[..., np.ndarray]


# List of transformations without string names
TRANSFORMATIONS: List[Tuple[TransformationFunction, ParameterGenerator]] = [
    (identity, lambda: {}),
    (rotate, lambda: {"angle": random.randint(-45, 45)}),
    (horizontal_flip, lambda: {}),
    (vertical_flip, lambda: {}),
    (
        crop,
        lambda: {
            "start_x": random.randint(0, 100),
            "start_y": random.randint(0, 100),
            "width": random.randint(100, 300),
            "height": random.randint(100, 300),
        },
    ),
    (adjust_brightness, lambda: {"value": random.randint(20, 100)}),
    (perspective_transform, lambda: {}),
    (
        random_erasing,
        lambda: {
            "probability": random.uniform(0.3, 0.7),
            "sl": random.uniform(0.02, 0.05),
            "sh": random.uniform(0.2, 0.5),
            "r1": random.uniform(0.2, 1.0),
        },
    ),
    (add_gaussian_noise, lambda: {}),
    (adjust_contrast, lambda: {"factor": random.uniform(0.8, 2.0)}),
]

# Example of how to use the TRANSFORMATIONS list
"""
Example usage of TRANSFORMATIONS:

>>> func, gen_kwargs = TRANSFORMATIONS[0]  # Access the first transformation, which is rotate
>>> kwargs = gen_kwargs()
>>> print(kwargs)  # Example output might be {'angle': -12}
>>> image = ...  # Your input image goes here
>>> transformed_image = func(image, **kwargs)  # Apply the transformation with generated parameters
"""
