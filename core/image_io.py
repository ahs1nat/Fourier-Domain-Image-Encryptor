import numpy as np

try:
    import cv2 as cv
except ImportError:
    cv = None

try:
    from PIL import Image
except ImportError:
    Image = None


def load_gray_img(image_path: str) -> np.ndarray:
    if cv is not None:
        img = cv.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at: {image_path}")
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_normalized = gray.astype(np.float64) / 255.0
        return gray_normalized
    elif Image is not None:
        img = Image.open(image_path).convert("L")
        gray_normalized = np.array(img, dtype=np.float64) / 255.0
        return gray_normalized
    else:
        raise ModuleNotFoundError(
            "Neither OpenCV (cv2) nor PIL (Pillow) is available to load images."
        )


def save_img_uint8(array: np.ndarray, output_path: str):
    clipped = np.clip(array, 0, 1)
    img_uint8 = (clipped * 255).astype(np.uint8)
    if cv is not None:
        cv.imwrite(output_path, img_uint8)
    elif Image is not None:
        Image.fromarray(img_uint8).save(output_path)
    else:
        raise ModuleNotFoundError(
            "Neither OpenCV (cv2) nor PIL (Pillow) is available to save images."
        )