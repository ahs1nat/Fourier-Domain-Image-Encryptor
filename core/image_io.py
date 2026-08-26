import cv2 as cv
import numpy as np

def load_gray_img(image_path: str) -> np.ndarray:
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_normalized = gray.astype(np.float64) / 255.0
    return gray_normalized

def save_img_uint8(array: np.ndarray, output_path: str):
    clipped = np.clip(array, 0, 1)
    img_uint8 = (clipped * 255).astype(np.uint8)
    cv.imwrite(output_path,img_uint8)