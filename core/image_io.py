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

def prepare_image_for_canvas(img: np.ndarray, canvas_shape: tuple = (512, 512)) -> tuple:
    """
    Fit any image onto a fixed canvas, centered:
    - if bigger than canvas: shrink to fit (aspect ratio preserved)
    - if smaller: center it with padding around all sides
    Works for grayscale (H, W) or RGB (H, W, 3).
    Returns (padded_image, (offset_y, offset_x, content_h, content_w))
    """
    canvas_h, canvas_w = canvas_shape
    is_rgb = img.ndim == 3
    h, w = img.shape[:2]

    if h > canvas_h or w > canvas_w:
        scale = min(canvas_h / h, canvas_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        pil_img = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img = np.asarray(pil_img, dtype=np.float64) / 255.0
        h, w = img.shape[:2]

    # Compute centering offsets
    offset_y = (canvas_h - h) // 2
    offset_x = (canvas_w - w) // 2

    if is_rgb:
        padded = np.zeros((canvas_h, canvas_w, 3), dtype=img.dtype)
        padded[offset_y:offset_y + h, offset_x:offset_x + w, :] = img
    else:
        padded = np.zeros((canvas_h, canvas_w), dtype=img.dtype)
        padded[offset_y:offset_y + h, offset_x:offset_x + w] = img

    return padded, (offset_y, offset_x, h, w)


def restore_original_size(decrypted: np.ndarray, content_info: tuple) -> np.ndarray:
    """Crop the decrypted canvas-sized image back to its real, centered content area."""
    offset_y, offset_x, h, w = content_info
    return decrypted[offset_y:offset_y + h, offset_x:offset_x + w]