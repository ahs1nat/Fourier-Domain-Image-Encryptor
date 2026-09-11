import os
import numpy as np
from core import (
    load_gray_img, save_img_uint8,
    prepare_image_for_canvas, restore_original_size,
    generate_key_pairs, save_library, load_library, get_pair,
    encrypt, decrypt,
)
from utils.metrics import image_score

CANVAS_SHAPE = (512, 512)


def main():
    os.makedirs("data/keys", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    input_path = "data/sample_input.png"
    library_path = "data/keys/test_library.pkl"
    cipher_path = "outputs/encrypted.png"
    shape_path = "outputs/original_shape.txt"
    decrypted_path = "outputs/decrypted.png"

    # 1. A loads the image and fits it onto the 512x512 canvas
    raw_image = load_gray_img(input_path)
    image, original_shape = prepare_image_for_canvas(raw_image, CANVAS_SHAPE)

    # Save original_shape so B knows how much to crop later
    with open(shape_path, "w") as f:
        f.write(f"{original_shape[0]},{original_shape[1]}")

    # 2. A creates/loads the 10-key library, sized to the canvas
    if os.path.exists(library_path):
        library = load_library(library_path)
    else:
        library = generate_key_pairs(n=10, canvas_shape=CANVAS_SHAPE)
        save_library(library, library_path)

    # 3. A picks one key and encrypts
    labels = list(library.keys())
    chosen_label = labels[3]
    P1, P2 = get_pair(library, chosen_label)
    ciphertext = encrypt(image, P1, P2)
    save_img_uint8(np.abs(ciphertext), cipher_path)
    print(f"A encrypted the image using: {chosen_label}")

    # 4. B tries every key pair and picks the most "real-looking" result
    best_label, best_score, best_image = None, float("inf"), None
    for label, pair in library.items():
        attempt = decrypt(ciphertext, pair["P1"], pair["P2"])
        score = image_score(attempt)
        print(f"Trying {label}: score = {score:.4f}")
        if score < best_score:
            best_label, best_score, best_image = label, score, attempt
            
    if best_image is None:
        raise RuntimeError("Brute force failed: key library was empty.")

    # 5. B crops the result back to the original image size
    with open(shape_path) as f:
        h, w = map(int, f.read().split(","))
    final_image = restore_original_size(best_image, (h, w))

    save_img_uint8(final_image, decrypted_path)
    print(f"\nB's best guess: {best_label}")
    print(f"Correct key was: {chosen_label}")
    print(f"Match: {best_label == chosen_label}")


if __name__ == "__main__":
    main()