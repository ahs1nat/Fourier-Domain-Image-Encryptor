r"""
Fourier-Domain Image Encryptor -- Streamlit Web Application
============================================================
Run with:
    .venv\Scripts\streamlit.exe run app.py
"""

import io
import pickle

import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from core import (
    crop_from_canvas,
    decrypt,
    decrypt_rgb,
    encrypt,
    encrypt_rgb,
    generate_phase_mask,
    pad_to_canvas,
)
from core.key_library import (
    load_library,
    load_library_encrypted,
    save_library_encrypted,
)
from core.metrics import (
    compute_psnr,
    compute_mse,
    compute_ssim,
    histogram_entropy,
    noise_robustness,
    histogram_comparison_image,
)

# ── page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fourier Image Encryptor",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #080f1a !important;
    font-family: "Inter", sans-serif;
}
[data-testid="stSidebar"] {
    background: rgba(13,27,42,0.95) !important;
    border-right: 1px solid rgba(0,245,212,0.15) !important;
    backdrop-filter: blur(20px);
}
h1 {
    background: linear-gradient(135deg,#00F5D4,#00B4D8,#FF006E);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
}
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,245,212,0.15);
    border-radius: 16px;
    padding: 1.4rem;
    backdrop-filter: blur(12px);
    margin-bottom: 0.9rem;
    transition: border-color .3s,box-shadow .3s;
}
.glass-card:hover {
    border-color: rgba(0,245,212,0.35);
    box-shadow: 0 0 24px rgba(0,245,212,0.08);
}
.metric-tile {
    background: linear-gradient(135deg,rgba(0,245,212,.08),rgba(0,180,216,.05));
    border: 1px solid rgba(0,245,212,.2);
    border-radius: 12px;
    padding: .9rem 1.1rem;
    text-align: center;
    margin-bottom: .7rem;
}
.metric-tile .label {font-size:.75rem;color:#8ecae6;text-transform:uppercase;letter-spacing:1px;}
.metric-tile .value {font-size:1.35rem;font-weight:700;color:#00F5D4;font-family:"JetBrains Mono",monospace;}
.metric-tile .value.warn {color:#FF006E;}
[data-testid="stTabs"] button {color:#8ecae6 !important;font-weight:500 !important;}
[data-testid="stTabs"] button[aria-selected="true"] {color:#00F5D4 !important;border-bottom:2px solid #00F5D4 !important;}
.stButton > button {
    background: linear-gradient(135deg,#00F5D4,#00B4D8) !important;
    color: #080f1a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 20px rgba(0,245,212,.25) !important;
    transition: opacity .2s,transform .15s,box-shadow .2s !important;
}
.stButton > button:hover {
    opacity: .88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(0,245,212,.4) !important;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,.03) !important;
    border: 1.5px dashed rgba(0,245,212,.25) !important;
    border-radius: 12px !important;
}
[data-testid="stDownloadButton"] > button {
    background: rgba(0,245,212,.1) !important;
    color: #00F5D4 !important;
    border: 1px solid rgba(0,245,212,.35) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(0,245,212,.18) !important;
    box-shadow: 0 0 14px rgba(0,245,212,.2) !important;
}
hr {border-color: rgba(0,245,212,.12) !important;}
::-webkit-scrollbar {width:6px;}
::-webkit-scrollbar-thumb {background:rgba(0,245,212,.2);border-radius:3px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── constants ──────────────────────────────────────────────────────────────────

CANVAS_CHOICES = {
    "256 x 256": (256, 256),
    "512 x 512": (512, 512),
    "1024 x 1024": (1024, 1024),
}

# ── helpers ────────────────────────────────────────────────────────────────────

def _to_gray(raw: bytes) -> np.ndarray:
    buf = np.frombuffer(raw, np.uint8)
    img = cv.imdecode(buf, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image. Use PNG, JPG, TIFF or BMP.")
    return img.astype(np.float64) / 255.0


def _to_rgb(raw: bytes) -> np.ndarray:
    buf = np.frombuffer(raw, np.uint8)
    img = cv.imdecode(buf, cv.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Use PNG, JPG, TIFF or BMP.")
    return cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float64) / 255.0


def _cmap(arr: np.ndarray, name: str = "gray") -> Image.Image:
    colormap = matplotlib.colormaps[name]
    rgba = colormap(np.clip(arr, 0, 1))
    return Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8))


def _gray_pil(arr: np.ndarray) -> Image.Image:
    return _cmap(arr, "gray")


def _rgb_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def _mag_pil(ct: np.ndarray) -> Image.Image:
    if ct.ndim == 3:
        mag = np.mean(np.abs(ct), axis=2)
    else:
        mag = np.abs(ct)
    lo, hi = mag.min(), mag.max()
    return _cmap((mag - lo) / (hi - lo + 1e-12), "inferno")


def _phase_heatmap(mask: np.ndarray, attr: str, cname: str) -> Image.Image:
    data = {"real": np.real, "imag": np.imag, "angle": np.angle}[attr](mask)
    lo, hi = data.min(), data.max()
    return _cmap((data - lo) / (hi - lo + 1e-12), cname)


def _to_npy(arr: np.ndarray) -> bytes:
    buf = io.BytesIO(); np.save(buf, arr); return buf.getvalue()


def _to_pkl(obj) -> bytes:
    buf = io.BytesIO(); pickle.dump(obj, buf); return buf.getvalue()


def _to_ekey(obj, password: str) -> bytes:
    import os, base64
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    salt = os.urandom(32)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480000)
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    token = Fernet(key).encrypt(pickle.dumps(obj))
    return salt + token


def _from_ekey(data: bytes, password: str) -> dict:
    import base64
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    salt, token = data[:32], data[32:]
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480000)
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    try:
        return pickle.loads(Fernet(key).decrypt(token))
    except Exception:
        raise ValueError("Incorrect password or corrupted key file.")


def _to_png(arr: np.ndarray, rgb: bool = False) -> bytes:
    if rgb:
        img_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        ok, enc = cv.imencode(".png", cv.cvtColor(img_u8, cv.COLOR_RGB2BGR))
    else:
        img_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        ok, enc = cv.imencode(".png", img_u8)
    if not ok:
        raise RuntimeError("PNG encoding failed.")
    return enc.tobytes()


def _metric_tile(label: str, value: str, warn: bool = False) -> str:
    cls = "value warn" if warn else "value"
    return (
        f"<div class='metric-tile'>"
        f"<div class='label'>{label}</div>"
        f"<div class='{cls}'>{value}</div>"
        f"</div>"
    )

# ── sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔐 DRPE Encryptor")
    st.markdown(
        "<div style='color:#8ecae6;font-size:.85rem;line-height:1.6;'>"
        "Double Random Phase Encoding (DRPE) encrypts images in the Fourier domain "
        "using two independent random phase masks <b>P\u2081</b> (spatial) and <b>P\u2082</b> (Fourier)."
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        "<div style='color:#8ecae6;font-size:.8rem;'>"
        "<b style='color:#00F5D4;'>Algorithm</b><br>"
        "Encrypt: <code>C = IFFT(FFT(I\u00b7P\u2081)\u00b7P\u2082)</code><br>"
        "Decrypt: <code>I = IFFT(FFT(C)\u00b7P\u2082*)\u00b7P\u2081*</code>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("v2.0 \u00b7 Built with Streamlit")

# ── header ─────────────────────────────────────────────────────────────────────

st.markdown("# 🔐 Fourier-Domain Image Encryptor")
st.markdown(
    "<p style='color:#8ecae6;font-size:1.05rem;margin-top:-.5rem;'>"
    "Optical-style encryption via Double Random Phase Encoding (DRPE)"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

tab_enc, tab_dec, tab_inspect = st.tabs(
    ["🔒  Encrypt", "🔓  Decrypt", "🔬  Phase Mask Inspector"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 -- ENCRYPT
# ══════════════════════════════════════════════════════════════════════════════

with tab_enc:
    st.markdown("### Upload an Image to Encrypt")
    col_cfg, col_prev = st.columns([1, 1], gap="large")

    with col_cfg:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        mode = st.radio("Colour mode", ["Grayscale", "RGB (colour)"], horizontal=True, key="enc_mode")
        uploaded_img = st.file_uploader(
            "Drop image here (PNG, JPG, TIFF, BMP)",
            type=["png", "jpg", "jpeg", "tiff", "bmp"],
            key="enc_uploader",
            label_visibility="collapsed",
        )
        canvas_label = st.selectbox(
            "Canvas size",
            list(CANVAS_CHOICES.keys()),
            index=1,
            key="enc_canvas",
            help="Image is zero-padded to this size. Must be >= image dimensions.",
        )
        canvas_shape = CANVAS_CHOICES[canvas_label]

        use_password = st.toggle("🔐 Password-protect key file (.ekey)", key="enc_pwd_toggle")
        password_enc = ""
        if use_password:
            password_enc = st.text_input(
                "Set encryption password",
                type="password",
                key="enc_pwd",
                placeholder="Enter a strong password...",
            )

        encrypt_btn = st.button("⚡ Generate Keys & Encrypt", key="enc_btn", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # preview
    raw_bytes = None
    with col_prev:
        if uploaded_img is not None:
            raw_bytes = uploaded_img.read()
            try:
                if mode == "RGB (colour)":
                    preview = _to_rgb(raw_bytes)
                    pil_prev = _rgb_pil(preview)
                else:
                    preview = _to_gray(raw_bytes)
                    pil_prev = _gray_pil(preview)
                h, w = preview.shape[:2]
                st.markdown(
                    _metric_tile("Image dimensions", f"{w} x {h} px"),
                    unsafe_allow_html=True,
                )
                st.image(pil_prev, caption="Loaded image preview", use_container_width=True)
            except ValueError as exc:
                st.error(str(exc))
                raw_bytes = None
        else:
            st.info("Upload an image on the left to preview it here.")

    # encrypt action
    if encrypt_btn:
        if raw_bytes is None:
            st.error("Please upload a valid image first.")
        elif use_password and not password_enc:
            st.error("Enter a password to protect your key file, or disable the toggle.")
        else:
            with st.spinner("Encrypting..."):
                if mode == "RGB (colour)":
                    img_data = _to_rgb(raw_bytes)
                    h0, w0 = img_data.shape[:2]
                else:
                    img_data = _to_gray(raw_bytes)
                    h0, w0 = img_data.shape

                if h0 > canvas_shape[0] or w0 > canvas_shape[1]:
                    st.error(
                        f"Image ({w0}x{h0}) exceeds the chosen canvas "
                        f"({canvas_shape[1]}x{canvas_shape[0]}). Choose a larger canvas."
                    )
                    st.stop()

                if mode == "RGB (colour)":
                    padded = np.zeros((*canvas_shape, 3), dtype=np.float64)
                    padded[:h0, :w0, :] = img_data
                    orig_shape = (h0, w0)
                    P1 = generate_phase_mask(canvas_shape)
                    P2 = generate_phase_mask(canvas_shape)
                    ciphertext = encrypt_rgb(padded, P1, P2)
                else:
                    padded, orig_shape = pad_to_canvas(img_data, canvas_shape)
                    P1 = generate_phase_mask(canvas_shape)
                    P2 = generate_phase_mask(canvas_shape)
                    ciphertext = encrypt(padded, P1, P2)

                st.session_state["enc_result"] = {
                    "ciphertext": ciphertext,
                    "P1": P1, "P2": P2,
                    "orig_shape": orig_shape,
                    "canvas_shape": canvas_shape,
                    "mode": mode,
                    "password": password_enc if use_password else None,
                }
            st.success("Encryption complete!")

    if "enc_result" in st.session_state:
        res = st.session_state["enc_result"]
        ct = res["ciphertext"]
        P1, P2 = res["P1"], res["P2"]
        orig_shape, canvas_shape_used = res["orig_shape"], res["canvas_shape"]
        enc_mode = res["mode"]
        pwd = res["password"]

        st.divider()
        st.markdown("### Results")
        col_a, col_b = st.columns(2, gap="medium")

        with col_a:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Ciphertext magnitude**")
            st.image(_mag_pil(ct), caption="Encrypted (inferno colormap)", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Encryption summary**")
            st.markdown(
                _metric_tile("Colour mode", enc_mode.split()[0])
                + _metric_tile("Canvas", f"{canvas_shape_used[1]}x{canvas_shape_used[0]}")
                + _metric_tile("Original crop", f"{orig_shape[1]}x{orig_shape[0]}")
                + _metric_tile("Key protected", "AES-256" if pwd else "None (plaintext)"),
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Download")
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Ciphertext (.npy)",
                data=_to_npy(ct),
                file_name="ciphertext.npy",
                mime="application/octet-stream",
                use_container_width=True,
            )
        with dl2:
            key_bundle = {
                "P1": P1, "P2": P2,
                "orig_shape": orig_shape,
                "canvas_shape": canvas_shape_used,
                "mode": enc_mode,
            }
            if pwd:
                st.download_button(
                    "⬇️ Encrypted key file (.ekey)",
                    data=_to_ekey(key_bundle, pwd),
                    file_name="drpe_keys.ekey",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
            else:
                st.download_button(
                    "⬇️ Key file (.pkl)",
                    data=_to_pkl(key_bundle),
                    file_name="drpe_keys.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 -- DECRYPT
# ══════════════════════════════════════════════════════════════════════════════

with tab_dec:
    st.markdown("### Decrypt an Encrypted Image")
    col_ul1, col_ul2 = st.columns(2, gap="large")

    with col_ul1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Step 1 — Ciphertext**")
        ct_file = st.file_uploader(
            "Ciphertext (.npy)",
            type=["npy"],
            key="dec_ct",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_ul2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Step 2 — Key file (.pkl or .ekey)**")
        key_file = st.file_uploader(
            "Key file",
            type=["pkl", "ekey"],
            key="dec_key",
            label_visibility="collapsed",
        )
        password_dec = ""
        if key_file is not None and key_file.name.endswith(".ekey"):
            password_dec = st.text_input(
                "Password for this .ekey file",
                type="password",
                key="dec_pwd",
                placeholder="Enter decryption password...",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Optional: upload original for metrics
    with st.expander("📊 Compare with original image (optional — for quality metrics)"):
        orig_file = st.file_uploader(
            "Original image (for PSNR/SSIM/MSE)",
            type=["png", "jpg", "jpeg", "tiff", "bmp"],
            key="dec_orig",
            label_visibility="collapsed",
        )

    decrypt_btn = st.button("🔓 Decrypt Image", key="dec_btn")

    if decrypt_btn:
        if ct_file is None or key_file is None:
            st.error("Upload both the ciphertext (.npy) and a key file (.pkl or .ekey).")
        elif key_file.name.endswith(".ekey") and not password_dec:
            st.error("Enter the password to unlock this .ekey file.")
        else:
            with st.spinner("Decrypting..."):
                try:
                    ct_arr = np.load(io.BytesIO(ct_file.read()), allow_pickle=False)
                    raw_key = key_file.read()

                    if key_file.name.endswith(".ekey"):
                        key_bundle = _from_ekey(raw_key, password_dec)
                    else:
                        key_bundle = pickle.loads(raw_key)

                    P1 = key_bundle["P1"]
                    P2 = key_bundle["P2"]
                    raw_shape = key_bundle.get("orig_shape")
                    orig_shape = raw_shape[:2] if raw_shape is not None else ct_arr.shape[:2]
                    dec_mode = key_bundle.get("mode", "Grayscale")

                    if dec_mode == "RGB (colour)":
                        dec_padded = decrypt_rgb(ct_arr, P1, P2)
                    else:
                        dec_padded = np.clip(decrypt(ct_arr, P1, P2), 0, 1)

                    decrypted = crop_from_canvas(dec_padded, orig_shape)

                    st.session_state["dec_result"] = {
                        "decrypted": decrypted,
                        "ciphertext": ct_arr,
                        "P1": P1, "P2": P2,
                        "mode": dec_mode,
                    }
                    st.session_state.pop("dec_metrics", None)
                except ValueError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Decryption failed: {exc}")

            if "dec_result" in st.session_state:
                st.success("Decryption complete!")

    if "dec_result" in st.session_state:
        res = st.session_state["dec_result"]
        decrypted = res["decrypted"]
        ct_arr = res["ciphertext"]
        dec_mode = res["mode"]
        is_rgb = dec_mode == "RGB (colour)"

        st.divider()
        st.markdown("### Results")
        col_ct, col_dec = st.columns(2, gap="medium")

        with col_ct:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Ciphertext magnitude**")
            st.image(_mag_pil(ct_arr), caption="Encrypted", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_dec:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Decrypted image**")
            dec_pil = _rgb_pil(decrypted) if is_rgb else _gray_pil(decrypted)
            st.image(dec_pil, caption="Recovered image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # download
        st.download_button(
            "⬇️ Download decrypted image (.png)",
            data=_to_png(decrypted, rgb=is_rgb),
            file_name="decrypted.png",
            mime="image/png",
        )

        # ── Quality metrics (if original was uploaded) ──────────────────────
        if orig_file is not None:
            orig_raw = orig_file.read()
            try:
                orig_arr = _to_gray(orig_raw) if not is_rgb else _to_rgb(orig_raw)

                # resize original to match decrypted if shapes differ
                if orig_arr.shape[:2] != decrypted.shape[:2]:
                    h, w = decrypted.shape[:2]
                    orig_arr = cv.resize(
                        orig_arr.astype(np.float32), (w, h), interpolation=cv.INTER_AREA
                    ).astype(np.float64)

                # for RGB, flatten to grayscale for PSNR/SSIM
                if is_rgb:
                    orig_flat = orig_arr.mean(axis=2)
                    dec_flat = decrypted.mean(axis=2)
                else:
                    orig_flat = orig_arr
                    dec_flat = decrypted

                p = compute_psnr(orig_flat, dec_flat)
                m = compute_mse(orig_flat, dec_flat)
                s = compute_ssim(orig_flat, dec_flat)
                e_orig = histogram_entropy(orig_flat)
                e_dec = histogram_entropy(dec_flat)

                st.divider()
                st.markdown("#### 📊 Quality Metrics")
                mc1, mc2, mc3, mc4 = st.columns(4)
                psnr_str = "∞ dB" if p == float("inf") else f"{p:.2f} dB"
                with mc1:
                    st.markdown(_metric_tile("PSNR", psnr_str), unsafe_allow_html=True)
                with mc2:
                    st.markdown(_metric_tile("MSE", f"{m:.2e}"), unsafe_allow_html=True)
                with mc3:
                    st.markdown(_metric_tile("SSIM", f"{s:.4f}"), unsafe_allow_html=True)
                with mc4:
                    st.markdown(_metric_tile("Entropy (orig)", f"{e_orig:.2f} bits"), unsafe_allow_html=True)

                # Histogram comparison
                st.markdown("**Pixel Histogram Comparison**")
                hist_img = histogram_comparison_image(orig_flat, dec_flat, ("Original", "Decrypted"))
                st.image(hist_img, use_container_width=True)

                # ── Noise robustness tester ─────────────────────────────────
                st.divider()
                with st.expander("🛡️ Noise Robustness Tester"):
                    st.markdown(
                        "<p style='color:#8ecae6;font-size:.9rem;'>Add noise to the ciphertext and measure how "
                        "much quality degrades on decryption. Tests the encryption sensitivity.</p>",
                        unsafe_allow_html=True,
                    )
                    if is_rgb:
                        st.info("Noise robustness test uses the first channel (R) for grayscale comparison.")

                    n_type = st.radio(
                        "Noise type",
                        ["Salt & Pepper", "Gaussian"],
                        horizontal=True,
                        key="noise_type",
                    )
                    n_level = st.slider(
                        "Noise level / std-dev",
                        min_value=0.01,
                        max_value=0.25,
                        value=0.05,
                        step=0.01,
                        key="noise_level",
                    )
                    run_noise_btn = st.button("▶ Run Test", key="noise_btn")

                    if run_noise_btn:
                        with st.spinner("Running noise test..."):
                            P1_r = res["P1"]
                            P2_r = res["P2"]
                            ct_single = ct_arr[:, :, 0] if is_rgb else ct_arr
                            orig_single = orig_flat
                            ntype_key = "salt_pepper" if n_type == "Salt & Pepper" else "gaussian"
                            nres = noise_robustness(
                                ct_single, P1_r, P2_r, orig_single,
                                noise_type=ntype_key, level=n_level,
                            )
                        st.session_state["noise_result"] = nres

                    if "noise_result" in st.session_state:
                        nr = st.session_state["noise_result"]
                        np_str = "∞ dB" if nr["psnr"] == float("inf") else f"{nr['psnr']:.2f} dB"
                        nc1, nc2, nc3 = st.columns(3)
                        with nc1:
                            st.markdown(_metric_tile("PSNR after noise", np_str, warn=True), unsafe_allow_html=True)
                        with nc2:
                            st.markdown(_metric_tile("MSE after noise", f"{nr['mse']:.2e}", warn=True), unsafe_allow_html=True)
                        with nc3:
                            st.markdown(_metric_tile("SSIM after noise", f"{nr['ssim']:.4f}", warn=True), unsafe_allow_html=True)
                        col_n1, col_n2 = st.columns(2)
                        with col_n1:
                            st.image(_gray_pil(orig_flat), caption="Original", use_container_width=True)
                        with col_n2:
                            st.image(_gray_pil(nr["noisy_decrypted"]), caption="Noisy-decrypted", use_container_width=True)

            except Exception as exc:
                st.warning(f"Could not compute metrics: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 -- PHASE MASK INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════

with tab_inspect:
    st.markdown("### Phase Mask Inspector")
    st.markdown(
        "<p style='color:#8ecae6;'>Visualise the real part, imaginary part, and phase angle of a "
        "random DRPE phase mask. Each pixel is e^{i*2pi*r}, r ~ Uniform[0,1).</p>",
        unsafe_allow_html=True,
    )
    col_cfg2, col_vis2 = st.columns([1, 2], gap="large")

    with col_cfg2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        insp_size = st.selectbox("Mask size", list(CANVAS_CHOICES.keys()), index=0, key="insp_size")
        insp_shape = CANVAS_CHOICES[insp_size]
        insp_id = st.selectbox("Which mask", ["P1 (spatial)", "P2 (Fourier)"], key="insp_id_select")
        gen_btn = st.button("Generate Mask", key="insp_btn", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_vis2:
        if gen_btn:
            with st.spinner("Generating..."):
                mask = generate_phase_mask(insp_shape)
                st.session_state["insp_mask"] = mask
                st.session_state["insp_shape"] = insp_shape
                st.session_state["insp_mask_label"] = insp_id

        if "insp_mask" in st.session_state:
            m = st.session_state["insp_mask"]
            shape_str = f"{st.session_state['insp_shape'][1]}x{st.session_state['insp_shape'][0]}"
            st.markdown(f"**{st.session_state.get('insp_mask_label', insp_id)} -- {shape_str}**")
            c1, c2, c3 = st.columns(3, gap="small")
            with c1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.image(_phase_heatmap(m, "real", "coolwarm"), caption="Real part", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.image(_phase_heatmap(m, "imag", "coolwarm"), caption="Imaginary part", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with c3:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.image(_phase_heatmap(m, "angle", "hsv"), caption="Phase angle", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            ent = histogram_entropy((np.angle(m) + np.pi) / (2 * np.pi))
            st.markdown(
                _metric_tile("Phase angle entropy", f"{ent:.4f} bits"),
                unsafe_allow_html=True,
            )
            st.download_button(
                "⬇️ Download mask (.npy)",
                data=_to_npy(m),
                file_name=f"phase_mask_{insp_shape[0]}x{insp_shape[1]}.npy",
                mime="application/octet-stream",
            )
        else:
            st.info("Click **Generate Mask** to create and visualise a phase mask.")
