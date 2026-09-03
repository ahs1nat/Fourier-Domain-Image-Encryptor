import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.key_library import generate_key_pairs, save_library, load_library, get_pair

canvas_shape = (512, 512)

pairs = generate_key_pairs(10, canvas_shape)
print("Generated labels:", list(pairs.keys()))

save_library(pairs, "data/keys/test_library.pkl")
loaded = load_library("data/keys/test_library.pkl")

label = list(loaded.keys())[0]
P1, P2 = get_pair(loaded, label)
print(f"Retrieved pair for {label}, shapes:", P1.shape, P2.shape)