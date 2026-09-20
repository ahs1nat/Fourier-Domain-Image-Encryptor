import streamlit as st
import os
import numpy as np
from PIL import Image
import random
import io
import tempfile

from core import (
    generate_key_pairs,
    get_pair,
    save_library_encrypted,
    load_library_encrypted,
    encrypt,
    decrypt,
    encrypt_rgb,
    decrypt_rgb,
)

from utils.metrics import compute_image_hash, image_score

# PAGE CONFIG

st.set_page_config(
    page_title="DRPE Image Encryptor",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CUSTOM CSS

st.markdown(
    """
    <style>
    * {
        font-family: "Segoe UI Emoji", "Noto Color Emoji", "Apple Color Emoji", "Segoe UI", sans-serif;
    }
    .stTextInput small {
        display: none !important;
    }
    .stApp { background: #0d1420; }
    .main .block-container {
        padding-top: 1rem; padding-left: 3rem; padding-right: 3rem; max-width: 1400px;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #172033 100%);
        border-right: 1px solid #263246;
    }
    section[data-testid="stSidebar"] * { color: #f8fafc; }
    .sidebar-logo { text-align: center; padding: 10px 0 25px 0; }
    .sidebar-logo-icon { font-size: 42px; }
    .sidebar-logo-title { font-size: 21px; font-weight: 700; margin-top: 5px; }
    .sidebar-logo-subtitle { font-size: 12px; color: #94a3b8 !important; margin-top: 4px; }

    .app-header, .section-card, .step-card {
        background: #172033; border: 1px solid #263246; border-radius: 18px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
    }
    .app-header { padding: 25px 30px; margin-bottom: 25px; }
    .section-card { padding: 25px; margin-bottom: 20px; }
    .step-card { padding: 20px; min-height: 115px; border-radius: 16px; }

    .app-title, .section-title, .step-title { color: #f8fafc; font-weight: 700; }
    .app-title { font-size: 32px; margin-bottom: 4px; }
    .section-title { font-size: 20px; margin-bottom: 6px; }
    .step-title { font-size: 16px; }

    .app-subtitle, .section-description, .step-text { color: #94a3b8; }
    .app-subtitle { font-size: 15px; }
    .section-description { font-size: 14px; margin-bottom: 20px; }
    .step-text { font-size: 13px; margin-top: 5px; }

    .step-number {
        display: inline-block; background: #00e0c6; color: #0d1420; border-radius: 50%;
        width: 30px; height: 30px; text-align: center; line-height: 30px;
        font-weight: 700; margin-bottom: 10px;
    }

    div[data-testid="stFileUploader"] { background: #172033; border-radius: 15px; }
    div[data-testid="stDownloadButton"] button { border-radius: 10px; font-weight: 600; }
    div.stButton > button { border-radius: 10px; font-weight: 600; min-height: 42px; }

    .status-card {
        padding: 15px 18px; border-radius: 12px; background: rgba(0,224,198,0.08);
        border: 1px solid rgba(0,224,198,0.3); color: #00e0c6; font-weight: 600; margin: 15px 0;
    }
    .formula-box {
        background: #0d1420; color: #00e0c6; border-radius: 14px; padding: 18px 22px;
        text-align: center; font-size: 18px; font-family: monospace; margin: 15px 0 25px 0;
        border: 1px solid #263246;
    }
    .footer { text-align: center; color: #64748b; font-size: 12px; padding: 35px 0 10px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# HELPER FUNCTIONS

def uploaded_file_to_array(file): # uploaded image k numpy file e convert kore
    img = Image.open(file)
    is_color = img.mode not in ("L", "1")
    img = img.convert("RGB") if is_color else img.convert("L")
    return np.asarray(img, dtype=np.float64) / 255.0, is_color


def normalize_for_display(array):
    """Convert a real/complex array into a displayable [0,1] image using magnitude."""
    magnitude = np.abs(array)
    min_val = magnitude.min()
    max_val = magnitude.max()
    return (magnitude - min_val) / (max_val - min_val + 1e-12)


def create_ciphertext_visualization(ciphertext, is_color):
    if is_color:
        magnitude = np.mean(np.abs(ciphertext), axis=2)
    else:
        magnitude = np.abs(ciphertext)
    min_val = magnitude.min()
    max_val = magnitude.max()
    return (magnitude - min_val) / (max_val - min_val + 1e-12)


def image_title(title, subtitle=None):
    subtitle_html = f'<div class="app-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-title">{title}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">🔐</div>
            <div class="sidebar-logo-title">DRPE Encryptor</div>
            <div class="sidebar-logo-subtitle">Fourier Domain Image Security</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🔒 Encrypt", "🔓 Decrypt"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="color:#94a3b8; font-size:12px; line-height:1.6;">
        <b>DRPE</b><br>
        Double Random Phase Encoding uses two random phase masks and
        Fourier transforms to transform an image into a ciphertext.
        </div>
        """,
        unsafe_allow_html=True,
    )


# MAIN HEADER

image_title(
    "Fourier Domain Image Encryptor",
    "Double Random Phase Encoding (DRPE) — secure image transformation in the Fourier domain."
)


# ENCRYPT PAGE

if page == "🔒 Encrypt":

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">🔒 Encrypt an Image</div>
            <div class="section-description">
                Upload an image, provide a password, and generate an encrypted
                ciphertext together with a password-protected key bundle.
            </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an image:",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        img_array, is_color = uploaded_file_to_array(uploaded_file)

        st.markdown("### Original Image")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(img_array, caption=f"Original image — {img_array.shape[:2]}", width="stretch")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            password = st.text_input(
                "🔑 Key file password",
                type="password",
                help="The receiver will need this password to open the .ekey file.",
            )
        with col2:
            st.info(
                "The password protects the DRPE key bundle. "
                "It is not used directly as the DRPE phase mask."
            )

        encrypt_btn = st.button("🔒 Encrypt Image", width="stretch")

        if encrypt_btn:
            if not password:
                st.error("Please enter a password before encrypting.")
            else:
                with st.spinner("Generating DRPE keys and encrypting image..."):
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

                    st.session_state["enc_result"] = {
                        "ciphertext": ciphertext,
                        "library": library,
                        "is_color": is_color,
                        "original_hash": original_hash,
                        "image_shape": image_shape,
                        "chosen_label": chosen_label,
                        "password": password,
                        "original_image": img_array,
                    }

                st.success("Image encrypted successfully.")

    # --------------------------------------------------------
    # ENCRYPTION RESULT
    # --------------------------------------------------------

    if "enc_result" in st.session_state:
        res = st.session_state["enc_result"]
        ciphertext = res["ciphertext"]
        is_color = res["is_color"]

        st.markdown("---")
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">📦 Encryption Result</div>
                <div class="section-description">
                    The image has been transformed into a Fourier-domain ciphertext.
                    The visualization below shows its magnitude.
                </div>
            """,
            unsafe_allow_html=True,
        )

        cipher_visual = create_ciphertext_visualization(ciphertext, is_color)

        col1, col2 = st.columns(2)
        with col1:
            st.image(res["original_image"], caption="Original image", width="stretch")
        with col2:
            st.image(cipher_visual, caption="Ciphertext magnitude visualization", width="stretch")

        st.markdown("</div>", unsafe_allow_html=True)

        # KEY INFORMATION
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">🔑 Key Information</div>
                <div class="section-description">
                    Ten random DRPE key pairs were generated. One was randomly
                    selected for encryption — B does not know which one.
                </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Key pairs generated", "10")
        with c2:
            st.metric("Image dimensions", f"{res['image_shape'][0]} × {res['image_shape'][1]}")
        with c3:
            st.metric("Image type", "RGB" if res["is_color"] else "Grayscale")

        with st.expander("🐛 Debug info (for your own verification only)"):
            st.write(f"Correct key: {res['chosen_label']}")

        st.markdown("</div>", unsafe_allow_html=True)

        # NOTE: chosen_label is intentionally excluded from full_bundle below —
        # B must not know which key was used ahead of time.
        full_bundle = {
            "library": res["library"],
            "is_color": res["is_color"],
            "original_hash": res["original_hash"],
            "image_shape": res["image_shape"],
        }

        with tempfile.NamedTemporaryFile(suffix=".ekey", delete=False) as tmp:
            temp_path = tmp.name

        save_library_encrypted(full_bundle, temp_path, res["password"])

        with open(temp_path, "rb") as f:
            ekey_data = f.read()
        os.remove(temp_path)

        npy_buf = io.BytesIO()
        np.save(npy_buf, ciphertext)
        npy_data = npy_buf.getvalue()

        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">📥 Download Files</div>
                <div class="section-description">
                    Send both files to the receiver. The receiver also needs
                    the password used for the .ekey file (shared separately).
                </div>
            """,
            unsafe_allow_html=True,
        )

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Download encrypted key (.ekey)",
                data=ekey_data,
                file_name="key_library.ekey",
                mime="application/octet-stream",
                width="stretch",
            )
        with dl2:
            st.download_button(
                "⬇️ Download ciphertext (.npy)",
                data=npy_data,
                file_name="ciphertext.npy",
                mime="application/octet-stream",
                width="stretch",
            )

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# DECRYPT PAGE
# ============================================================

elif page == "🔓 Decrypt":

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">🔓 Receiver / B</div>
            <div class="section-description">
                Upload the ciphertext and the encrypted key bundle, enter the
                password, and the system will try every phase mask pair —
                exactly one of them will recover the correct image.
            </div>
        """,
        unsafe_allow_html=True,
    )

    ciphertext_file = st.file_uploader(
        "Upload ciphertext (.npy)",
        type=["npy"],
        key="ciphertext_upload",
    )

    ekey_file = st.file_uploader(
        "Upload encrypted key (.ekey)",
        type=["ekey"],
        key="ekey_upload",
    )

    password = st.text_input(
        "🔑 Key file password",
        type="password",
        key="decrypt_password",
    )

    decrypt_btn = st.button(
        "🔓 Decrypt Image",
        width="stretch",
        disabled=not (ciphertext_file and ekey_file and password),
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if decrypt_btn:
        if ekey_file is None or ciphertext_file is None:
            st.error("Please upload both the ciphertext (.npy) and the key file (.ekey) before decrypting.")
        else:
            with st.spinner("Unlocking key file and trying all phase mask pairs..."):
                with tempfile.NamedTemporaryFile(suffix=".ekey", delete=False) as tmp:
                    tmp.write(ekey_file.getvalue())
                    temp_path = tmp.name

                try:
                    bundle = load_library_encrypted(temp_path, password)
                except ValueError as e:
                    st.error(str(e))
                    bundle = None
                finally:
                    os.remove(temp_path)

                if bundle is not None:
                    library = bundle["library"]
                    is_color = bundle["is_color"]
                    original_hash = bundle["original_hash"]

                    ciphertext = np.load(io.BytesIO(ciphertext_file.getvalue()))

                    results = []
                    for label, pair in library.items():
                        if is_color:
                            attempt = decrypt_rgb(ciphertext, pair["P1"], pair["P2"])
                            score = image_score(attempt.mean(axis=2))
                        else:
                            attempt = decrypt(ciphertext, pair["P1"], pair["P2"])
                            score = image_score(attempt)

                        attempt_hash = compute_image_hash(attempt)
                        is_match = (attempt_hash == original_hash)

                        results.append({
                            "label": label,
                            "image": attempt,
                            "score": score,
                            "match": is_match,
                        })

                    st.session_state["dec_results"] = results

    # --------------------------------------------------------
    # DECRYPTION RESULTS
    # --------------------------------------------------------

    if "dec_results" in st.session_state:
        results = st.session_state["dec_results"]
        correct = next((r for r in results if r["match"]), None)

        st.markdown("---")

        if correct:
            st.markdown(
                f"""
                <div class="status-card">
                    ✅ Correct key found: {correct['label']} (verified by hash match)
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("No exact hash match found among the keys tried.")

        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">🔍 All Attempts</div>
                <div class="section-description">
                    Every phase mask pair was tried. The image score indicates
                    how "real" the result looks — the hash match gives a
                    definitive verification.
                </div>
            """,
            unsafe_allow_html=True,
        )

        cols = st.columns(5)
        for i, r in enumerate(results):
            with cols[i % 5]:
                st.image(r["image"], caption=r["label"], width="stretch")
                st.markdown(
                    f"Score: `{r['score']:.4f}` &nbsp; "
                    f"{'✅' if r['match'] else '❌'}"
                )

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        DRPE Image Encryptor · Fourier Domain Image Processing
    </div>
    """,
    unsafe_allow_html=True,
)