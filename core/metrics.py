import hashlib
import numpy as np

def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Squared Error between two normalised [0, 1] arrays."""
    return float(np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB).  Returns inf for identical images."""
    mse_val = mse(original, reconstructed)
    if mse_val < 1e-20:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse_val))


def image_score(image: np.ndarray) -> float:
    """
    Simple 'does this look like a real image or noise' score, used for
    brute-force key guessing. Real images are smooth (low pixel-to-pixel
    change); noise is not. Lower score = more likely to be the correct key.
    """
    diff_x = np.diff(image, axis=1)
    diff_y = np.diff(image, axis=0)
    return float(np.mean(np.abs(diff_x)) + np.mean(np.abs(diff_y)))


def compute_image_hash(image: np.ndarray) -> str:
    """
    Compute a SHA-256 hash of an image's pixel data, used to verify
    whether a decryption attempt exactly matches the original image.
    Quantizes to uint8 first so tiny floating-point differences from
    FFT math don't cause false mismatches.
    """
    img_bytes = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8).tobytes()
    return hashlib.sha256(img_bytes).hexdigest()


def adjacent_pixel_correlation(img: np.ndarray, direction: str = "horizontal", n_samples: int = 3000) -> tuple:
    """
    Sample pairs of adjacent pixel values from an image, for correlation analysis.
    Real images: pairs cluster tightly (neighboring pixels are similar).
    Random noise: pairs scatter with no structure.

    Parameters
    ----------
    img : 2D array (grayscale)
    direction : "horizontal", "vertical", or "diagonal"
    n_samples : number of random pixel pairs to sample (for plotting speed)

    Returns
    -------
    (x_vals, y_vals, correlation_coefficient)
    """
    h, w = img.shape

    if direction == "horizontal":
        x = img[:, :-1]
        y = img[:, 1:]
    elif direction == "vertical":
        x = img[:-1, :]
        y = img[1:, :]
    elif direction == "diagonal":
        x = img[:-1, :-1]
        y = img[1:, 1:]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    x_flat = x.ravel()
    y_flat = y.ravel()

    rng = np.random.default_rng(42)
    if len(x_flat) > n_samples:
        idx = rng.choice(len(x_flat), size=n_samples, replace=False)
        x_flat = x_flat[idx]
        y_flat = y_flat[idx]

    if x_flat.std() == 0 or y_flat.std() == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(x_flat, y_flat)[0, 1])

    return x_flat, y_flat, corr
