"""
core/metrics.py
===============
Image quality and security evaluation metrics for DRPE-encrypted images.
"""
import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Squared Error between two normalised [0, 1] arrays."""
    return float(np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB).  Returns inf for identical images."""
    mse_val = mse(original, reconstructed)
    if mse_val < 1e-20:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse_val))


def ssim(original: np.ndarray, reconstructed: np.ndarray, data_range: float = 1.0) -> float:
    """
    Global Structural Similarity Index (SSIM).
    Computed over the full image (not windowed), yielding a single scalar.
    Range [-1, 1]; 1 = identical.
    """
    o = original.astype(np.float64)
    r = reconstructed.astype(np.float64)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu1, mu2 = o.mean(), r.mean()
    sigma1_sq = o.var()
    sigma2_sq = r.var()
    sigma12 = float(np.mean((o - mu1) * (r - mu2)))
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(numerator / denominator)


# Aliases
compute_mse = mse
compute_psnr = psnr
compute_ssim = ssim



def histogram_entropy(img: np.ndarray, bins: int = 256) -> float:
    """Shannon entropy (bits) of the pixel-value distribution."""
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8).ravel()
    counts, _ = np.histogram(img_uint8, bins=bins, range=(0, 255))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def add_salt_pepper_noise(img: np.ndarray, level: float) -> np.ndarray:
    """Salt-and-pepper noise at density *level* in [0, 1]."""
    noisy = img.copy()
    rng = np.random.default_rng()
    mask = rng.random(img.shape)
    noisy[mask < level / 2] = 0.0
    noisy[mask > 1 - level / 2] = 1.0
    return noisy


def add_gaussian_noise(img: np.ndarray, std: float) -> np.ndarray:
    """Zero-mean Gaussian noise with standard deviation *std*."""
    rng = np.random.default_rng()
    return np.clip(img + rng.normal(0, std, img.shape), 0, 1)


def noise_robustness(
    ciphertext: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    original: np.ndarray,
    noise_type: str = "salt_pepper",
    level: float = 0.05,
) -> dict:
    """
    Corrupt the ciphertext magnitude with noise, decrypt, and return quality metrics.

    Parameters
    ----------
    noise_type : {"salt_pepper", "gaussian"}
    level      : noise density (salt_pepper) or std dev (gaussian)

    Returns
    -------
    dict with keys: psnr, mse, ssim, noisy_decrypted (ndarray)
    """
    from .drpe import decrypt  # relative import avoids circular dependency

    mag = np.abs(ciphertext)
    phase = np.angle(ciphertext)
    lo, hi = mag.min(), mag.max()
    mag_norm = (mag - lo) / (hi - lo + 1e-12)

    if noise_type == "salt_pepper":
        mag_noisy_norm = add_salt_pepper_noise(mag_norm, level)
    else:
        mag_noisy_norm = add_gaussian_noise(mag_norm, level)

    mag_noisy = mag_noisy_norm * (hi - lo) + lo
    ct_noisy = mag_noisy * np.exp(1j * phase)

    from .drpe import decrypt as _decrypt
    dec = _decrypt(ct_noisy, P1, P2)
    dec = np.clip(dec, 0, 1)

    return {
        "psnr": psnr(original, dec),
        "mse": mse(original, dec),
        "ssim": ssim(original, dec),
        "noisy_decrypted": dec,
    }


def histogram_comparison_image(
    img1: np.ndarray,
    img2: np.ndarray,
    labels: tuple = ("Original", "Decrypted"),
) -> Image.Image:
    """Render a side-by-side pixel histogram comparison as a PIL Image."""
    matplotlib.use("Agg")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), facecolor="#0d1b2a")
    colors = ["#00F5D4", "#FF006E"]
    for ax, img, label, color in zip(axes, [img1, img2], labels, colors):
        vals = (np.clip(img, 0, 1) * 255).astype(np.uint8).ravel()
        ax.hist(vals, bins=64, color=color, alpha=0.85, edgecolor="none")
        ax.set_title(label, color="white", fontsize=11, fontweight="bold")
        ax.set_facecolor("#0d1b2a")
        ax.tick_params(colors="#8ecae6", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1a3a5c")
        ax.set_xlabel("Pixel value", color="#8ecae6", fontsize=8)
        ax.set_ylabel("Count", color="#8ecae6", fontsize=8)
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()
