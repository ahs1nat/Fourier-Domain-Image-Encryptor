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


def prepare_image_for_canvas(img: np.ndarray, canvas_shape: tuple = (512, 512)) -> tuple:
    """
    Prepares any input image to fit into a fixed canvas size (default 512x512).

    - If the image is LARGER than canvas_shape in either dimension,
      it is smart-resized (aspect-ratio preserved, shrunk to fit).
    - If the image is SMALLER than canvas_shape, it is placed onto
      a zero-padded canvas of exactly canvas_shape (no stretching).
    - If it already matches canvas_shape, it's used as-is.

    Returns:
        (padded_image, original_shape_before_padding)
        original_shape_before_padding is needed later to crop back
        to the true content size after decryption.
    """
    canvas_h, canvas_w = canvas_shape
    h, w = img.shape

    # Case 1: image too big in either dimension -> resize down, preserving aspect ratio
    if h > canvas_h or w > canvas_w:
        scale = min(canvas_h / h, canvas_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
        h, w = img.shape

    # Case 2 & 3: pad (or exact fit, padding does nothing) onto canvas
    padded = np.zeros(canvas_shape, dtype=img.dtype)
    padded[:h, :w] = img

    return padded, (h, w)


def restore_original_size(decrypted: np.ndarray, original_shape: tuple) -> np.ndarray:
    """
    Crops the decrypted 512x512 image back down to the content region
    that was placed there before encryption (removes the zero padding).
    """
    h, w = original_shape
    return decrypted[:h, :w]