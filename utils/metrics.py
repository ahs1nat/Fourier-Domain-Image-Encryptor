import numpy as np

def image_score(image: np.ndarray) -> float:
    """
    Simple way to check if an image looks 'real' vs random noise.
    Real images have smooth areas -> low pixel-to-pixel variation.
    Noise has no structure -> high pixel-to-pixel variation.
    Lower score = more likely to be the correct decryption.
    """
    diff_x = np.diff(image, axis=1)
    diff_y = np.diff(image, axis=0)
    return float(np.mean(np.abs(diff_x)) + np.mean(np.abs(diff_y)))