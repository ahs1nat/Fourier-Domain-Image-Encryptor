import pickle
import string
import secrets
from core.drpe import generate_phase_mask

def generate_label() -> str:
    alphabet = string.ascii_uppercase + string.digits
    part1 = "".join(secrets.choice(alphabet) for _ in range(4))
    part2 = "".join(secrets.choice(alphabet) for _ in range(4))
    return f"KEY-{part1}-{part2}"

def generate_key_pairs(n: int, canvas_shape: tuple) -> dict:
    """
    Generate n random (P1, P2) phase mask pairs, each sized to canvas_shape,
    each identified by a unique random label.

    Returns:
        dict: { label: {"P1": ndarray, "P2": ndarray} }
    """
    pairs = {}
    labels_seen = set()

    while len(pairs) < n:
        label = generate_label()
        if label in labels_seen:
            continue  # extremely unlikely collision
        labels_seen.add(label)

        P1 = generate_phase_mask(canvas_shape)
        P2 = generate_phase_mask(canvas_shape)
        pairs[label] = {"P1": P1, "P2": P2}

    return pairs

def save_library(pairs_dict: dict, path: str) -> None:
    """
    Save the key-library dict to disk as a pickle file.
    This is the file A and B exchange securely beforehand.
    """
    with open(path, "wb") as f:
        pickle.dump(pairs_dict, f)


def load_library(path: str) -> dict:
    """
    Load a previously saved key-library file.
    """
    with open(path, "rb") as f:
        pairs_dict = pickle.load(f)
    return pairs_dict


def get_pair(pairs_dict: dict, label: str) -> tuple:
    """
    Retrieve (P1, P2) for a given label.
    Raises KeyError with a clear message if label doesn't exist.
    """
    if label not in pairs_dict:
        raise KeyError(f"Label '{label}' not found in key library.")
    entry = pairs_dict[label]
    return entry["P1"], entry["P2"]
