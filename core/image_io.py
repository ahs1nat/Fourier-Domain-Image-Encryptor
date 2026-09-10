import cv2 as cv
import numpy as np


def load_gray_img(image_path: str) -> np.ndarray:
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_normalized = gray.astype(np.float64) / 255.0
    return gray_normalized


def save_img_uint8(array: np.ndarray, output_path: str) -> None:
    clipped = np.clip(array, 0, 1)
    img_uint8 = (clipped * 255).astype(np.uint8)
    cv.imwrite(output_path, img_uint8)


def pad_to_canvas(img: np.ndarray, canvas_shape: tuple) -> tuple:
    h, w = img.shape
    canvas_h, canvas_w = canvas_shape

    if h > canvas_h or w > canvas_w:
        raise ValueError(
            f"Image shape {img.shape} exceeds canvas shape {canvas_shape}. "
            f"Choose a larger canvas size."
        )

    padded = np.zeros(canvas_shape, dtype=img.dtype)
    padded[:h, :w] = img
    return padded, (h, w)


def crop_from_canvas(img: np.ndarray, original_shape: tuple) -> np.ndarray:
    h, w = original_shape
    return img[:h, :w]


def load_color_img(image_path: str) -> np.ndarray:
    """
    Load an image and return a normalised float64 RGB array shaped (H, W, 3)
    with values in [0, 1].
    """
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return rgb.astype(np.float64) / 255.0