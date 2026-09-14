import os
import numpy as np
from core import (
    load_color_img, save_img_uint8,
    generate_key_pairs, save_library, load_library, get_pair,
    encrypt_rgb, decrypt_rgb,
)
from utils.metrics import image_score


def main():
    os.makedirs("data/keys", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    input_path = "data/sample_input.jpg"
    library_path = "data/keys/test_library.pkl"
    cipher_path = "outputs/encrypted.png"
    decrypted_path = "outputs/decrypted.png"

    # 1. A loads the image as RGB (used as-is, no resizing/padding)
    image = load_color_img(input_path)  # shape (H, W, 3)
    image_shape = image.shape[:2]  # only (H, W) needed for mask generation

    # 2. A creates/loads the 10-key library, sized to this image's H,W
    if os.path.exists(library_path):
        library = load_library(library_path)
    else:
        library = generate_key_pairs(n=10, canvas_shape=image_shape)
        save_library(library, library_path)

    # 3. A picks one key and encrypts
    labels = list(library.keys())
    chosen_label = labels[3]
    P1, P2 = get_pair(library, chosen_label)
    ciphertext = encrypt_rgb(image, P1, P2)

    cipher_mag = np.mean(np.abs(ciphertext), axis=2)  # flatten for visualization
    save_img_uint8(cipher_mag / cipher_mag.max(), cipher_path)
    print(f"A encrypted the image using: {chosen_label}")

    # 4. B tries every key pair and picks the most "real-looking" result
    best_label, best_score, best_image = None, float("inf"), None
    for label, pair in library.items():
        attempt = decrypt_rgb(ciphertext, pair["P1"], pair["P2"])
        score = image_score(attempt.mean(axis=2))  # flatten to grayscale just for scoring
        print(f"Trying {label}: score = {score:.4f}")
        if score < best_score:
            best_label, best_score, best_image = label, score, attempt

    if best_image is None:
        raise RuntimeError("Brute force failed: key library was empty.")

    # 5. Save B's best guess directly — still full RGB
    save_img_uint8(best_image, decrypted_path)
    print(f"\nB's best guess: {best_label}")
    print(f"Correct key was: {chosen_label}")
    print(f"Match: {best_label == chosen_label}")


if __name__ == "__main__":
    main()