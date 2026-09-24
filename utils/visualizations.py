"""
utils/visualizations.py
======================
Visualization helper functions for plotting metrics, histograms,
phase spectra, key comparison scores, and intermediate DRPE pipeline stages.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from core.metrics import adjacent_pixel_correlation
from core.drpe import decrypt


def create_histogram_fig(original: np.ndarray, ciphertext: np.ndarray, decrypted: np.ndarray | None = None):
    """
    Generate a 3-panel pixel intensity histogram comparing original, ciphertext, and decrypted images.
    """
    fig, axes = plt.subplots(1, 3 if decrypted is not None else 2, figsize=(14, 3.8), dpi=100)
    fig.patch.set_facecolor('#0E1117')

    # Convert to 1D flat arrays
    orig_flat = original.ravel()
    
    # Ciphertext magnitude normalized
    if ciphertext.ndim == 3:
        c_mag = np.mean(np.abs(ciphertext), axis=2)
    else:
        c_mag = np.abs(ciphertext)
    c_norm = (c_mag - c_mag.min()) / (c_mag.max() - c_mag.min() + 1e-12)
    c_flat = c_norm.ravel()

    # Original Histogram
    axes[0].hist(orig_flat, bins=50, color='#00D2FF', alpha=0.85, edgecolor='black', linewidth=0.3)
    axes[0].set_title("Original Image Intensity", color='#E0E0E0', fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Pixel Value [0, 1]", color='#AAAAAA', fontsize=9)
    axes[0].set_ylabel("Pixel Count", color='#AAAAAA', fontsize=9)
    axes[0].set_facecolor('#161B22')
    axes[0].tick_params(colors='#AAAAAA', labelsize=8)
    axes[0].grid(True, linestyle='--', alpha=0.2, color='#555555')

    # Ciphertext Histogram
    axes[1].hist(c_flat, bins=50, color='#FF007A', alpha=0.85, edgecolor='black', linewidth=0.3)
    axes[1].set_title("Ciphertext Intensity (White Noise)", color='#E0E0E0', fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Normalized Magnitude [0, 1]", color='#AAAAAA', fontsize=9)
    axes[1].set_facecolor('#161B22')
    axes[1].tick_params(colors='#AAAAAA', labelsize=8)
    axes[1].grid(True, linestyle='--', alpha=0.2, color='#555555')

    # Decrypted Histogram
    if decrypted is not None:
        dec_flat = decrypted.ravel()
        axes[2].hist(dec_flat, bins=50, color='#00FF87', alpha=0.85, edgecolor='black', linewidth=0.3)
        axes[2].set_title("Decrypted Image Intensity", color='#E0E0E0', fontsize=11, fontweight='bold')
        axes[2].set_xlabel("Pixel Value [0, 1]", color='#AAAAAA', fontsize=9)
        axes[2].set_facecolor('#161B22')
        axes[2].tick_params(colors='#AAAAAA', labelsize=8)
        axes[2].grid(True, linestyle='--', alpha=0.2, color='#555555')

    plt.tight_layout()
    return fig


def create_phase_spectrum_fig(P1: np.ndarray, P2: np.ndarray):
    """
    Generate 2D phase maps and 1D angle distribution histograms for P1 and P2.
    """
    angle_P1 = np.angle(P1) % (2 * np.pi)
    angle_P2 = np.angle(P2) % (2 * np.pi)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
    fig.patch.set_facecolor('#0E1117')

    # 2D Spatial Phase Map P1
    im0 = axes[0, 0].imshow(angle_P1, cmap='twilight', vmin=0, vmax=2*np.pi)
    axes[0, 0].set_title("Spatial Phase Mask P1(x, y)", color='#E0E0E0', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    cbar0 = fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cbar0.ax.tick_params(colors='#AAAAAA', labelsize=8)

    # Angle Histogram P1
    axes[0, 1].hist(angle_P1.ravel(), bins=60, color='#7928CA', alpha=0.85, edgecolor='black', linewidth=0.3)
    axes[0, 1].set_title("P1 Phase Distribution [0, 2π]", color='#E0E0E0', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Phase Angle (radians)", color='#AAAAAA', fontsize=9)
    axes[0, 1].set_ylabel("Count", color='#AAAAAA', fontsize=9)
    axes[0, 1].set_facecolor('#161B22')
    axes[0, 1].tick_params(colors='#AAAAAA', labelsize=8)
    axes[0, 1].grid(True, linestyle='--', alpha=0.2, color='#555555')

    # 2D Fourier Phase Map P2
    im1 = axes[1, 0].imshow(angle_P2, cmap='twilight', vmin=0, vmax=2*np.pi)
    axes[1, 0].set_title("Fourier Spectral Phase Mask P2(u, v)", color='#E0E0E0', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    cbar1 = fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(colors='#AAAAAA', labelsize=8)

    # Angle Histogram P2
    axes[1, 1].hist(angle_P2.ravel(), bins=60, color='#0070F3', alpha=0.85, edgecolor='black', linewidth=0.3)
    axes[1, 1].set_title("P2 Phase Distribution [0, 2π]", color='#E0E0E0', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("Phase Angle (radians)", color='#AAAAAA', fontsize=9)
    axes[1, 1].set_ylabel("Count", color='#AAAAAA', fontsize=9)
    axes[1, 1].set_facecolor('#161B22')
    axes[1, 1].tick_params(colors='#AAAAAA', labelsize=8)
    axes[1, 1].grid(True, linestyle='--', alpha=0.2, color='#555555')

    plt.tight_layout()
    return fig


def create_key_scores_fig(labels: list, scores: list, correct_label: str | None = None):
    """
    Bar chart showing smoothness/error score across all tried keys during brute-force decryption.
    """
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=100)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#161B22')

    colors = []
    for label in labels:
        if label == correct_label:
            colors.append('#00FF87')  # Bright green for matched key
        else:
            colors.append('#333333')  # Dark gray for wrong keys

    bars = ax.bar(labels, scores, color=colors, edgecolor='#555555', linewidth=0.6, width=0.55)
    
    ax.set_title("Brute-Force Key Smoothness Score (Lower = Real Image Match)", color='#E0E0E0', fontsize=11, fontweight='bold')
    ax.set_ylabel("Image Gradient Score", color='#AAAAAA', fontsize=9)
    ax.set_xlabel("Key Identifier", color='#AAAAAA', fontsize=9)
    ax.tick_params(colors='#AAAAAA', labelsize=8)
    plt.xticks(rotation=35, ha='right')
    ax.grid(True, linestyle='--', alpha=0.2, color='#555555', axis='y')

    # Highlight correct key bar value
    if correct_label in labels and len(scores) > 0:
        idx = labels.index(correct_label)
        ax.text(idx, scores[idx] + max(scores)*0.03, "✓ MATCH", ha='center', va='bottom', color='#00FF87', fontweight='bold', fontsize=9)

    plt.tight_layout()
    return fig


def generate_drpe_intermediate_stages(image: np.ndarray, P1: np.ndarray, P2: np.ndarray, is_color: bool = False):
    """
    Computes all intermediate visual stages of the DRPE 4-f optical correlator pipeline.
    """
    if is_color:
        img_gray = np.mean(image, axis=2)
    else:
        img_gray = image.copy()

    # Helper to scale array to [0, 1]
    def norm(arr):
        a_min, a_max = arr.min(), arr.max()
        return (arr - a_min) / (a_max - a_min + 1e-12)

    # Stage 1: Input image
    stage1 = np.clip(img_gray, 0, 1)

    # Stage 2: Spatial modulation (I * P1)
    step1_complex = img_gray * P1
    stage2_mag = norm(np.abs(step1_complex))
    stage2_phase = (np.angle(step1_complex) % (2 * np.pi)) / (2 * np.pi)

    # Stage 3: 2D FFT Spectrum
    step2_fft = np.fft.fftshift(np.fft.fft2(step1_complex))
    stage3_mag_log = norm(np.log(1.0 + np.abs(step2_fft)))
    stage3_phase = (np.angle(step2_fft) % (2 * np.pi)) / (2 * np.pi)

    # Stage 4: Spectral modulation (* P2)
    P2_shifted = np.fft.fftshift(P2)
    step3_modulated = step2_fft * P2_shifted
    stage4_mag_log = norm(np.log(1.0 + np.abs(step3_modulated)))
    stage4_phase = (np.angle(step3_modulated) % (2 * np.pi)) / (2 * np.pi)

    # Stage 5: Ciphertext (2D IFFT)
    ciphertext_complex = np.fft.ifft2(np.fft.ifftshift(step3_modulated))
    c_mag = np.abs(ciphertext_complex)
    stage5_mag_norm = norm(c_mag)
    stage5_phase = (np.angle(ciphertext_complex) % (2 * np.pi)) / (2 * np.pi)

    # Stage 6: Decryption with correct vs wrong key
    stage6_correct = norm(np.abs(decrypt(ciphertext_complex, P1, P2)))
    P2_wrong = np.exp(1j * 2 * np.pi * np.random.rand(*image.shape[:2]))
    stage6_wrong = norm(np.abs(decrypt(ciphertext_complex, P1, P2_wrong)))

    return {
        "stage1_input": stage1,
        "stage2_mag": stage2_mag,
        "stage2_phase": stage2_phase,
        "stage3_mag_log": stage3_mag_log,
        "stage3_phase": stage3_phase,
        "stage4_mag_log": stage4_mag_log,
        "stage4_phase": stage4_phase,
        "stage5_cipher_mag": stage5_mag_norm,
        "stage5_cipher_phase": stage5_phase,
        "stage6_correct": stage6_correct,
        "stage6_wrong": stage6_wrong,
    }


def create_correlation_scatter_fig(original: np.ndarray, ciphertext: np.ndarray, direction: str = "horizontal"):
    """
    Side-by-side scatter plots of adjacent-pixel correlation for the original
    image vs the ciphertext magnitude. Real images cluster along a diagonal;
    properly encrypted ciphertext should scatter randomly.
    """
    c_mag = np.abs(ciphertext)
    c_norm = (c_mag - c_mag.min()) / (c_mag.max() - c_mag.min() + 1e-12)

    ox, oy, o_corr = adjacent_pixel_correlation(original, direction)
    cx, cy, c_corr = adjacent_pixel_correlation(c_norm, direction)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=100)
    fig.patch.set_facecolor('#0E1117')

    axes[0].scatter(ox, oy, s=4, color='#00D2FF', alpha=0.5, edgecolors='none')
    axes[0].set_title(f"Original Image  (r = {o_corr:.3f})", color='#E0E0E0', fontsize=11, fontweight='bold')
    axes[0].set_xlabel(f"Pixel value at (x, y)", color='#AAAAAA', fontsize=9)
    axes[0].set_ylabel(f"Pixel value at neighbor ({direction})", color='#AAAAAA', fontsize=9)
    axes[0].set_facecolor('#161B22')
    axes[0].tick_params(colors='#AAAAAA', labelsize=8)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, linestyle='--', alpha=0.2, color='#555555')

    axes[1].scatter(cx, cy, s=4, color='#FF007A', alpha=0.5, edgecolors='none')
    axes[1].set_title(f"Ciphertext  (r = {c_corr:.3f})", color='#E0E0E0', fontsize=11, fontweight='bold')
    axes[1].set_xlabel(f"Pixel value at (x, y)", color='#AAAAAA', fontsize=9)
    axes[1].set_ylabel(f"Pixel value at neighbor ({direction})", color='#AAAAAA', fontsize=9)
    axes[1].set_facecolor('#161B22')
    axes[1].tick_params(colors='#AAAAAA', labelsize=8)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, linestyle='--', alpha=0.2, color='#555555')

    plt.tight_layout()
    return fig