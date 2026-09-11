import numpy as np

def generate_phase_mask(shape: tuple) -> np.ndarray: # P1
    random_phase = np.random.rand(*shape) # * unpacks the shape tuple
    # all values in random_phase are between 0 (inclusive) and 1 (exclusive)

    phase_mask = np.exp(1j * 2 * np.pi * random_phase)

    return phase_mask

def encrypt(image: np.ndarray, P1: np.ndarray, P2:np.ndarray) -> np.ndarray:
    if image.shape != P1.shape or image.shape != P2.shape:
        raise ValueError(
            f"Shape mismatch: image {image.shape}, P1 {P1.shape}, P2 {P2.shape}"
        )
    step1 = image * P1
    step2 = np.fft.fft2(step1)
    step3 = step2 * P2
    ciphertext = np.fft.ifft2(step3)
    return ciphertext

def decrypt(ciphertext: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    if ciphertext.shape != P1.shape or ciphertext.shape != P2.shape:
        raise ValueError(
            f"Shape mismatch: ciphertext {ciphertext.shape}, P1 {P1.shape}, P2 {P2.shape}"
        )
    step1 = np.fft.fft2(ciphertext)
    step2 = step1 * np.conj(P2)
    step3 = np.fft.ifft2(step2)
    decrypted = step3 * np.conj(P1)
    return np.abs(decrypted)


def encrypt_rgb(image_rgb: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    Encrypt a normalised RGB image [H, W, 3] using DRPE applied per channel.

    Parameters
    ----------
    image_rgb : float64 array shaped (H, W, 3), values in [0, 1]
    P1, P2    : phase masks shaped (H, W)

    Returns
    -------
    complex128 array shaped (H, W, 3)
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got {image_rgb.shape}")
    return np.stack([encrypt(image_rgb[:, :, c], P1, P2) for c in range(3)], axis=2)


def decrypt_rgb(ciphertext_rgb: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    Decrypt a complex RGB ciphertext [H, W, 3] using DRPE applied per channel.

    Returns
    -------
    float64 array shaped (H, W, 3), clipped to [0, 1]
    """
    if ciphertext_rgb.ndim != 3 or ciphertext_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB ciphertext (H, W, 3), got {ciphertext_rgb.shape}")
    return np.clip(
        np.stack([decrypt(ciphertext_rgb[:, :, c], P1, P2) for c in range(3)], axis=2),
        0, 1,
    )