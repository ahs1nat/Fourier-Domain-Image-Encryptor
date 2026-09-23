import unittest
import numpy as np
from core.drpe import generate_phase_mask, encrypt, decrypt


class TestDRPE(unittest.TestCase):
    def test_drpe_encryption_decryption_reconstruction(self):
        shape = (128, 128)
        image = np.random.rand(*shape)
        P1 = generate_phase_mask(shape)
        P2 = generate_phase_mask(shape)

        ciphertext = encrypt(image, P1, P2)
        decrypted_image = decrypt(ciphertext, P1, P2)

        # Reconstructed image should closely match original image
        np.testing.assert_allclose(image, decrypted_image, atol=1e-12)

    def test_drpe_decryption_wrong_fourier_key_fails(self):
        shape = (64, 64)
        image = np.random.rand(*shape)
        P1 = generate_phase_mask(shape)
        P2 = generate_phase_mask(shape)
        P2_wrong = generate_phase_mask(shape)

        ciphertext = encrypt(image, P1, P2)
        decrypted_image = decrypt(ciphertext, P1, P2_wrong)

        # Decrypting with wrong Fourier key P2 should yield noise, not original image
        self.assertFalse(np.allclose(image, decrypted_image, atol=1e-2))

    def test_drpe_decryption_shape_mismatch(self):
        image = np.random.rand(32, 32)
        P1 = generate_phase_mask((32, 32))
        P2 = generate_phase_mask((32, 32))
        ciphertext = encrypt(image, P1, P2)

        P_wrong_shape = generate_phase_mask((16, 16))
        with self.assertRaises(ValueError):
            decrypt(ciphertext, P_wrong_shape, P2)


    def test_image_hash_match_discrete_pixels(self):
        from core.metrics import compute_image_hash
        shape = (64, 64)
        # Create discrete 8-bit normalized image
        image = np.round(np.random.rand(*shape) * 255) / 255.0
        P1 = generate_phase_mask(shape)
        P2 = generate_phase_mask(shape)

        ciphertext = encrypt(image, P1, P2)
        decrypted_image = decrypt(ciphertext, P1, P2)

        self.assertEqual(compute_image_hash(image), compute_image_hash(decrypted_image))


if __name__ == "__main__":
    unittest.main()


