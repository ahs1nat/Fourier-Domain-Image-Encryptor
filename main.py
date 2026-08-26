import os
import numpy as np
from core import load_gray_img, save_img_uint8, generate_phase_mask, encrypt, decrypt


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    input_path = os.path.join("data", "sample_input.png")
    cipher_path = os.path.join("outputs", "encrypted.png")
    decrypted_path = os.path.join("outputs", "decrypted.png")

    # 1. Create a sample test image if input.png does not exist
    if not os.path.exists(input_path):
        print(f"Creating a sample image at {input_path}...")
        sample_img = np.zeros((256, 256), dtype=np.uint8)
        # Draw a square and a circle pattern
        sample_img[64:192, 64:192] = 180
        y, x = np.ogrid[:256, :256]
        mask = (x - 128) ** 2 + (y - 128) ** 2 <= 40**2
        sample_img[mask] = 255
        save_img_uint8(sample_img / 255.0, input_path)

    # 2. Load input image as normalized grayscale array [0, 1]
    print(f"Loading image from {input_path}...")
    image = load_gray_img(input_path)

    # 3. Generate phase mask keys (P1 and P2)
    print("Generating random phase masks (P1, P2)...")
    P1 = generate_phase_mask(image.shape)
    P2 = generate_phase_mask(image.shape)

    # 4. Encrypt the image
    print("Encrypting image with DRPE...")
    ciphertext = encrypt(image, P1, P2)

    # Save visualization of ciphertext (magnitude normalized)
    cipher_magnitude = np.abs(ciphertext)
    cipher_normalized = (cipher_magnitude - cipher_magnitude.min()) / (
        cipher_magnitude.max() - cipher_magnitude.min() + 1e-12
    )
    save_img_uint8(cipher_normalized, cipher_path)
    print(f"Encrypted ciphertext visualization saved to {cipher_path}")

    # 5. Decrypt using the same phase keys
    print("Decrypting ciphertext with original keys (P1, P2)...")
    decrypted_image = decrypt(ciphertext, P1, P2)

    # Save reconstructed decrypted image
    save_img_uint8(decrypted_image, decrypted_path)
    print(f"Decrypted image saved to {decrypted_path}")

    # Calculate reconstruction error
    diff = np.max(np.abs(image - decrypted_image))
    print(f"Max absolute difference between original and decrypted: {diff:.2e}")
    print("Done!")


if __name__ == "__main__":
    main()
