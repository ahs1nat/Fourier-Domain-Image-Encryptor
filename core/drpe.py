import numpy as np

def generate_phase_mask(shape: tuple) -> np.ndarray: # P1
    random_phase = np.random.rand(*shape) # * unpacks the shape tuple
    # all values in randomm_phase are between 0 (inclusive) and 1 (exclusive)

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