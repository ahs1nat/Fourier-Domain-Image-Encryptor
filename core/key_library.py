import os
import base64
import pickle
import string
import secrets
from .drpe import generate_phase_mask

# ── AES-256 encrypted key storage constants ───────────────────────────────────
_SALT_BYTES = 32
_PBKDF2_ITERATIONS = 480_000

def generate_label() -> str: # P1 P2 er jonno identifier
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
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
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
    if label not in pairs_dict:
        raise KeyError(f"Label '{label}' not found in key library.")
    entry = pairs_dict[label]
    """
    {
    "P1": ...,
    "P2": ...
    }
    """
    return entry["P1"], entry["P2"]


# ── Encrypted key storage ─────────────────────────────────────────────────────

def _derive_fernet_key(password: str, salt: bytes) -> bytes:
    """Derive a 32-byte Fernet key from *password* + *salt* via PBKDF2-HMAC-SHA256."""
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=_PBKDF2_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))

# Password + Random salt -> PBKDF2 -> 32-byte key -> Fernet key

def save_library_encrypted(bundle: dict, path: str, password: str) -> None:
    """
    Encrypt *pairs_dict* with AES-256 (Fernet) and save to *path* (.ekey).

    File format: 32-byte random salt | Fernet token
    Key derivation: PBKDF2-HMAC-SHA256 with 480 000 iterations.
    """
    from cryptography.fernet import Fernet

    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    salt = os.urandom(_SALT_BYTES)
    fernet_key = _derive_fernet_key(password, salt)
    token = Fernet(fernet_key).encrypt(pickle.dumps(bundle))
    with open(path, "wb") as fp:
        fp.write(salt + token)


def load_library_encrypted(path: str, password: str) -> dict:
    """
    Decrypt and return a key library previously saved with save_library_encrypted.

    Raises
    ------
    ValueError
        If the password is wrong or the file is corrupted.
    """
    from cryptography.fernet import Fernet, InvalidToken

    with open(path, "rb") as fp:
        data = fp.read()

    salt, token = data[:_SALT_BYTES], data[_SALT_BYTES:]
    fernet_key = _derive_fernet_key(password, salt)
    try:
        payload = Fernet(fernet_key).decrypt(token)
    except (InvalidToken, Exception) as exc:
        raise ValueError("Incorrect password or corrupted key file.") from exc
    return pickle.loads(payload)