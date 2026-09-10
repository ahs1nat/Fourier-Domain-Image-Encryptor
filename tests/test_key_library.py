import os
import sys
import unittest
import tempfile

from core.key_library import (
    generate_key_pairs,
    save_library,
    load_library,
    get_pair,
    generate_label,
)


class TestKeyLibrary(unittest.TestCase):
    def test_generate_label_format(self):
        label = generate_label()
        self.assertTrue(label.startswith("KEY-"))
        parts = label.split("-")
        self.assertEqual(len(parts), 3)

    def test_generate_and_save_load_library(self):
        canvas_shape = (128, 128)
        num_pairs = 5
        pairs = generate_key_pairs(num_pairs, canvas_shape)
        self.assertEqual(len(pairs), num_pairs)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "keys", "test_library.pkl")
            save_library(pairs, file_path)

            loaded = load_library(file_path)
            self.assertEqual(set(pairs.keys()), set(loaded.keys()))

            first_label = list(loaded.keys())[0]
            P1, P2 = get_pair(loaded, first_label)
            self.assertEqual(P1.shape, canvas_shape)
            self.assertEqual(P2.shape, canvas_shape)

    def test_get_pair_invalid_label_raises_key_error(self):
        pairs = generate_key_pairs(1, (64, 64))
        with self.assertRaises(KeyError):
            get_pair(pairs, "NON_EXISTENT_KEY")


if __name__ == "__main__":
    unittest.main()