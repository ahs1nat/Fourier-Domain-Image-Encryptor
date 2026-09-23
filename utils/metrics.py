"""
utils/metrics.py
================
Minimal image quality metrics for the DRPE project.
"""
import hashlib
import numpy as np


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Squared Error between two [0, 1] images."""
    return float(np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB). Higher = closer match. Inf if identical."""
    mse_val = mse(original, reconstructed)
    if mse_val < 1e-20:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse_val))


def image_score(image: np.ndarray) -> float:
    """
    Simple 'does this look like a real image or noise' score, used for
    brute-force key guessing. Real images are smooth (low pixel-to-pixel
    change); noise is not. Lower score = more likely to be correct.
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