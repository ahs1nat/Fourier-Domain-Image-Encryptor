import streamlit as st
import os
import numpy as np
from PIL import Image
import random
import pickle
import io

from core import (
    generate_key_pairs, save_library, load_library, get_pair,
    encrypt, decrypt, encrypt_rgb, decrypt_rgb,
    prepare_image_for_canvas, restore_original_size
)

from utils.metrics import image_score, compute_image_hash

st.set_page_config(page_title="Fourier Domain Image Encryptor", page_icon="🔐", layout="wide")

st.markdown( # tab center-align korar jonno
    """
    <style>
    div[data-testid="stTabs"] div[role="tablist"] {
        display: flex !important;
        justify-content: center !important;
        gap: 3rem !important;
    }
    button[data-baseweb="tab"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# CANVAS_SHAPE = (512, 512)
LIBRARY_PATH = os.path.join("data", "keys", "shared_library.pkl")

def uploaded_file_to_array(file):
    img = Image.open(file)
    is_color = img.mode not in ("L", "1")
    img = img.convert("RGB") if is_color else img.convert("L")
    return np.asarray(img, dtype=np.float64) / 255, is_color

st.title("🔐 Fourier Domain Image Encryptor")
st.caption("Double Random Phase Encoding (DRPE) — A encrypts with 1 of 10 keys, B brute-forces to find it.")

tab_a, tab_b = st.tabs(["🔒 A — Encrypt", "🔓 B — Decrypt"])

with tab_a:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        uploaded_file = st.file_uploader(label="Choose an image to encrypt:", type=["png", "jpg", "jpeg"], width=512, accept_multiple_files=False)
    if uploaded_file is not None:
        img_array, is_color = uploaded_file_to_array(uploaded_file)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.image(img_array, caption=f"Uploaded image ({img_array.shape[:2]})")
            encrypt_btn = st.button("🔒 Encrypt", width="stretch")

        if encrypt_btn:

            image_shape = img_array.shape[:2]
            library = generate_key_pairs(n=10, canvas_shape=image_shape)
            labels = list(library.keys())
            chosen_label = random.choice(labels)

            P1, P2 = get_pair(library, chosen_label)
            if is_color:
                ciphertext = encrypt_rgb(img_array, P1, P2)
            else:
                ciphertext = encrypt(img_array, P1, P2)

            original_hash = compute_image_hash(img_array)

            st.session_state["enc_result"] = { # rerun e jate variable gula reset na hoy
                "ciphertext": ciphertext,
                "library": library,
                "is_color": is_color,
                "original_hash": original_hash,
                "image_shape": image_shape,
                "chosen_label": chosen_label,
            }

        if "enc_result" in st.session_state:
            res = st.session_state["enc_result"]
            ciphertext = res["ciphertext"]
            is_color = res["is_color"]

            # ciphertext ta image hishebe show korar jonno
            cipher_mag = np.abs(ciphertext) if not is_color else np.mean(np.abs(ciphertext), axis=2)
            cipher_norm = (cipher_mag - cipher_mag.min()) / (cipher_mag.max() - cipher_mag.min() + 1e-12)

            st.markdown("Ciphertext (what an attacker without the key sees)")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.image(cipher_norm, caption="Encrypted image (magnitude visualization)", width=512)

            key_bundle = {
                "library": res["library"],
                "is_color": res["is_color"],
                "original_hash": res["original_hash"],
                "image_shape": res["image_shape"],
            }

            pkl_buf = io.BytesIO() # an empty virtual file that lives only in the RAM
            pickle.dump(key_bundle, pkl_buf)
            pkl_buf.seek(0) # pointer k shamne ane

            npy_buf = io.BytesIO()
            np.save(npy_buf, ciphertext)
            npy_buf.seek(0)

            st.success("Image encrypted!")

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇️ Key library (.pkl) — send to B beforehand",
                    data=pkl_buf,
                    file_name="key_library.pkl",
                    mime="application/octet-stream", # MIME is a label that tells the browser what kind of data this file contains
                )
            with dl2:
                st.download_button(
                    "⬇️ Encrypted image (.npy) — send to B when ready",
                    data=npy_buf,
                    file_name="ciphertext.npy",
                    mime="application/octet-stream",
                )
    

with tab_b:
    st.write("B's side goes here")