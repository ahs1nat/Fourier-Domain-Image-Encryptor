import numpy as np
from PIL import Image

RESAMPLE_FILTER = Image.Resampling.LANCZOS

def load_gray_img(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    return np.asarray(img, dtype=np.float64) / 255.0


def load_color_img(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def save_img_uint8(array: np.ndarray, output_path: str) -> None:
    clipped = np.clip(array, 0, 1)
    img_uint8 = (clipped * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(output_path)
