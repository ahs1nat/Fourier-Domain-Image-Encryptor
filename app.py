r"""
Fourier-Domain Image Encryptor -- Streamlit Web Application
============================================================
Run with:
    .venv\Scripts\streamlit.exe run app.py
"""

import io
import os
import random
import tempfile

import numpy as np
import streamlit as st
from PIL import Image

from core import encrypt, decrypt, encrypt_rgb, decrypt_rgb, generate_phase_mask
from core.key_library import (
    generate_key_pairs, get_pair,
    save_library_encrypted, load_library_encrypted,
)
from core.metrics import image_score, compute_image_hash, psnr, mse
from utils.visualizations import (
    create_histogram_fig,
    create_phase_spectrum_fig,
    create_key_scores_fig,
    generate_drpe_intermediate_stages,
)

st.set_page_config(
    page_title="Fourier Domain Image Encryptor",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Glassmorphism CSS Theme
st.markdown(
    """
    <style>
    * {
    font-family: "Inter",
                 "Segoe UI Emoji",
                 "Noto Color Emoji",
                 "Apple Color Emoji",
                 "Segoe UI",
                 sans-serif;
    }
    /* Global Page Styling */
    .stApp {
        background: linear-gradient(135deg, #0b0d12 0%, #161b26 50%, #0d111a 100%);
        color: #E6EDF3;
    }

    /* Header Styling */
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #00D2FF 0%, #3A7BD5 50%, #00FF87 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }

    .sub-header {
        text-align: center;
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }

    /* Center and Style Navigation Tabs */
    div[data-testid="stTabs"] div[role="tablist"] {
        display: flex !important;
        justify-content: center !important;
        gap: 1.5rem !important;
        background: rgba(22, 27, 38, 0.6);
        padding: 0.6rem 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
    }

    button[data-baseweb="tab"] {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.3s ease !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #0070F3 0%, #7928CA 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px 0 rgba(0, 118, 255, 0.39) !important;
    }

    /* Custom Glass Cards */
    .glass-card {
        background: rgba(22, 27, 38, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        margin-bottom: 1.5rem;
    }

    /* Metrics Badge Box */
    .metric-box {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Custom File Uploader & Buttons */
    .stButton>button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.3);
    }

    </style>
    """,
    unsafe_allow_html=True,
)
# st.divider()

# App Header
st.markdown('<div class="main-header">🔐 Fourier Domain Image Encryptor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Optical Cryptography Suite — Double Random Phase Encoding (DRPE) in the 4-f Correlator Plane</div>', unsafe_allow_html=True)


def uploaded_file_to_array(file):
    img = Image.open(file)
    is_color = img.mode not in ("L", "1")
    img = img.convert("RGB") if is_color else img.convert("L")
    return np.asarray(img, dtype=np.float64) / 255.0, is_color


# Tab Setup
tab_a, tab_b, tab_c, tab_d = st.tabs([
    "🔒 Encrypt",
    "🔓 Decrypt",
    "🔍 Phase Mask Inspector",
    "📘 How DRPE Works"
])

# ==============================================================================
# TAB A: ENCRYPTION
# ==============================================================================
with tab_a:
    st.markdown("<h3 style='text-align: center;'>🔒 Encrypt Image with DRPE Key Library</h3>", unsafe_allow_html=True)
    st.caption(
        "<p style='text-align: center;'>Upload an image to generate a 10-key phase mask library ($P_1, P_2$) and produce a noise-like ciphertext.</p>",
            unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        uploaded_file = st.file_uploader(
        label="Choose an image to encrypt:",
        type=["png", "jpg", "jpeg"]
    )
    #uploaded_file = st.file_uploader(label="Choose an image to encrypt:", type=["png", "jpg", "jpeg"])

    if uploaded_file is None:
        st.session_state.pop("enc_result", None)

    img_array = None
    is_color = False
    password = ""
    encrypt_btn = False

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        #st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        if uploaded_file is not None:
            img_array, is_color = uploaded_file_to_array(uploaded_file)
            st.markdown("##### Uploaded image")
            st.image(img_array, caption=f"Original Image ({img_array.shape[1]}x{img_array.shape[0]} | {'RGB' if is_color else 'Grayscale'})", width='stretch')

            password = st.text_input("🔑 Password for Key File (.ekey):", type="password", help="The recipient will require this password to unlock the .ekey file.")
            encrypt_btn = st.button("🔒 Execute DRPE Encryption", width='stretch', disabled=not password)
        #st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if uploaded_file is not None and img_array is not None and encrypt_btn:
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
                "P1": P1,
                "P2": P2
            }

        if "enc_result" in st.session_state:
            res = st.session_state["enc_result"]
            ciphertext = res["ciphertext"]
            is_color = res["is_color"]
            orig_img = res["original_image"]

            cipher_mag = np.abs(ciphertext) if not is_color else np.mean(np.abs(ciphertext), axis=2)
            cipher_norm = (cipher_mag - cipher_mag.min()) / (cipher_mag.max() - cipher_mag.min() + 1e-12)

            #st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            #st.write("")
            st.markdown("##### Ciphertext Output")
            st.image(cipher_norm, caption="Encrypted Magnitude Spectrum (Noise-like Ciphertext)", width='stretch')

            with tempfile.NamedTemporaryFile(suffix=".ekey", delete=False) as tmp:
                temp_path = tmp.name

            full_bundle = {
                "library": res["library"],
                "is_color": res["is_color"],
                "original_hash": res["original_hash"],
                "image_shape": res["image_shape"],
            }
            save_library_encrypted(full_bundle, temp_path, res["password"])

            with open(temp_path, "rb") as f:
                ekey_buf = io.BytesIO(f.read())
            os.remove(temp_path)

            ekey_buf.seek(0)
            npy_buf = io.BytesIO()
            np.save(npy_buf, ciphertext)
            npy_buf.seek(0)

            st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

            st.success("✅ Encryption Complete!")

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇️ Key Library (.ekey)",
                    data=ekey_buf,
                    file_name="key_library.ekey",
                    mime="application/octet-stream",
                    width='stretch',
                )
            with dl2:
                st.download_button(
                    "⬇️ Ciphertext (.npy)",
                    data=npy_buf,
                    file_name="ciphertext.npy",
                    mime="application/octet-stream",
                    width='stretch',
                )
            st.markdown('</div>', unsafe_allow_html=True)

    # Pixel Intensity Histogram Analysis
    if "enc_result" in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Pixel Distribution Analysis")
        res = st.session_state["enc_result"]
        fig_hist = create_histogram_fig(res["original_image"], res["ciphertext"])
        st.pyplot(fig_hist)


# ==============================================================================
# TAB B: DECRYPTION & BRUTE-FORCE
# ==============================================================================
with tab_b:
    st.markdown("<h3 style='text-align: center;'>🔓 Decrypt Ciphertext & Key Library Brute-Force</h3>", unsafe_allow_html=True)
    st.caption(
        "<p style='text-align: center;'>Upload the `.ekey` library and `.npy` ciphertext to unlock the key bundle and identify the matching phase mask.</p>",
        unsafe_allow_html=True
    )

    left, center, right = st.columns([1, 2, 1])
    with center:
        ekey_file = st.file_uploader("Upload Key Library (.ekey):", type=["ekey"])
        npy_file = st.file_uploader("Upload Ciphertext (.npy):", type=["npy"])

        if ekey_file is None or npy_file is None:
            st.session_state.pop("dec_result", None)

        dec_password = ""
        decrypt_btn = False

        if ekey_file and npy_file:
            dec_password = st.text_input("🔑 Password to unlock .ekey:", type="password")
            decrypt_btn = st.button("🔓 Decrypt & Brute-Force Keys", width='stretch', disabled=not dec_password)

    progress_bar = st.empty()
    status_text = st.empty()

    if ekey_file and npy_file and decrypt_btn:
        with tempfile.NamedTemporaryFile(suffix=".ekey", delete=False) as tmp:
            tmp.write(ekey_file.read())
            temp_ekey_path = tmp.name

        try:
            bundle = load_library_encrypted(temp_ekey_path, dec_password)
            ciphertext = np.load(npy_file)
            library = bundle["library"]
            is_color = bundle["is_color"]
            original_hash = bundle["original_hash"]

            labels = list(library.keys())
            pbar = progress_bar.progress(0)

            correct_label = None
            best_decrypted = None
            attempts = []

            for i, label in enumerate(labels):
                P1, P2 = get_pair(library, label)
                decrypted = decrypt_rgb(ciphertext, P1, P2) if is_color else decrypt(ciphertext, P1, P2)

                score = image_score(decrypted)
                current_hash = compute_image_hash(decrypted)
                is_match = current_hash == original_hash

                attempts.append({
                    "label": label,
                    "score": score,
                    "decrypted": decrypted,
                    "is_match": is_match,
                })

                if is_match:
                    correct_label = label
                    best_decrypted = decrypted

                status_text.text(f"Testing key {i+1}/{len(labels)} ({label})... Score: {score:.4f}")
                pbar.progress((i + 1) / len(labels))

            os.remove(temp_ekey_path)

            if correct_label and best_decrypted is not None:
                status_text.success(f"🎯 Match found: {correct_label} (tested all {len(labels)} keys)")
                st.session_state["dec_result"] = {
                    "decrypted": best_decrypted,
                    "ciphertext": ciphertext,
                    "correct_label": correct_label,
                    "attempts": attempts,
                    "original_hash": original_hash,
                }
            else:
                st.error("❌ No key in this library reconstructs the ciphertext.")

        except ValueError:
            if os.path.exists(temp_ekey_path):
                os.remove(temp_ekey_path)
            st.error("❌ Incorrect password or corrupted .ekey file!")

    # ==========================================================================
    # RESULT — centered, below everything above
    # ==========================================================================
    if "dec_result" in st.session_state:
        res_dec = st.session_state["dec_result"]

        result_left, result_center, result_right = st.columns([1, 2, 1])
        with result_center:
            st.image(
                res_dec["decrypted"],
                caption=f"Decrypted with correct key: {res_dec['correct_label']}",
                width='stretch',
            )
            dec_img_uint8 = (np.clip(res_dec["decrypted"], 0, 1) * 255).astype(np.uint8)
            pil_img = Image.fromarray(dec_img_uint8)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            buf.seek(0)

            st.download_button(
                label="⬇️ Download Decrypted Image",
                data=buf,
                file_name=f"decrypted_img.png",
                mime="image/png",
                width='stretch',
            )

        # ======================================================================
        # ALL 10 ATTEMPTS — scrollable row
        # ======================================================================
        attempts = res_dec["attempts"]

        st.markdown("---")
        st.markdown("<h4 style='text-align: center;'>🧪 All Key Attempts</h4>", unsafe_allow_html=True)
        st.caption(
            f"<p style='text-align: center;'>{len(attempts)} key(s) tried before a match was found (or exhausted).</p>",
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <style>
            .attempts-scroll-row {
                display: flex;
                overflow-x: auto;
                gap: 1rem;
                padding: 0.5rem 0 1rem 0;
            }
            .attempts-scroll-row > div {
                flex: 0 0 auto;
                width: 180px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        cols_per_row = 5
        for row_start in range(0, len(attempts), cols_per_row):
            row_attempts = attempts[row_start: row_start + cols_per_row]
            grid_cols = st.columns(len(row_attempts))
            for col, attempt in zip(grid_cols, row_attempts):
                with col:
                    caption = f"{attempt['label']}\nscore: {attempt['score']:.3f}"
                    if attempt["is_match"]:
                        caption = "✅ " + caption
                    st.image(attempt["decrypted"], caption=caption, width='stretch')

        # ======================================================================
        # ANALYSIS — centered, below attempts
        # ======================================================================
        st.markdown("---")
        st.markdown("<h4 style='text-align: center;'>📊 Key Attempt Analysis</h4>", unsafe_allow_html=True)

        labels_list = [a["label"] for a in attempts]
        scores_list = [a["score"] for a in attempts]

        analysis_left, analysis_center, analysis_right = st.columns([1, 3, 1])
        with analysis_center:
            fig_scores = create_key_scores_fig(
                labels_list, scores_list, correct_label=res_dec["correct_label"]
            )
            st.pyplot(fig_scores)

            if "enc_result" in st.session_state:
                st.markdown(
                    "<p style='text-align: center;'>Pixel Intensity Comparison (Original vs Ciphertext vs Decrypted)</p>",
                    unsafe_allow_html=True
                )
                fig_hist3 = create_histogram_fig(
                    st.session_state["enc_result"]["original_image"], res_dec["ciphertext"], res_dec["decrypted"]
                )
                st.pyplot(fig_hist3)


# ==============================================================================
# TAB C: PHASE MASK INSPECTOR
# ==============================================================================
with tab_c:
    st.markdown("### 🔍 Phase Mask Inspection & Randomness Analysis")
    st.caption("Inspect the spatial phase mask ($P_1$) and Fourier spectral phase mask ($P_2$) unit circle distributions.")

    if "enc_result" in st.session_state:
        P1 = st.session_state["enc_result"]["P1"]
        P2 = st.session_state["enc_result"]["P2"]
        fig_phase = create_phase_spectrum_fig(P1, P2)
        st.pyplot(fig_phase)
    else:
        # Generate demo masks for visual inspection
        demo_shape = (256, 256)
        P1_demo = generate_phase_mask(demo_shape)
        P2_demo = generate_phase_mask(demo_shape)
        st.info("Showing generated demo phase masks (encrypt an image in Tab A to inspect your custom key pair):")
        fig_phase = create_phase_spectrum_fig(P1_demo, P2_demo)
        st.pyplot(fig_phase)


# ==============================================================================
# TAB D: HOW DRPE WORKS (VISUAL WALKTHROUGH)
# ==============================================================================
with tab_d:
    st.markdown("### 📘 Double Random Phase Encoding (DRPE) — Step-by-Step Procedure")
    st.caption("Double Random Phase Encoding simulates an optical 4-f correlator system using 2D Fast Fourier Transforms (FFT).")

    st.markdown("""
    ```text
    [ Input Image I(x,y) ]
             │
             ▼
     [ × Spatial Mask P1(x,y) ]   ──> Phase modulation: P1 = exp(j * 2π * r1)
             │
             ▼
        [ 2D FFT ]                ──> Transform to 2D Spatial Frequency Domain
             │
             ▼
    [ × Spectral Mask P2(u,v) ]   ──> Spectral Modulation: P2 = exp(j * 2π * r2)
             │
             ▼
       [ 2D IFFT ]                ──> Inverse Fourier Transform
             │
             ▼
    [ Ciphertext C(x,y) ]         ──> Complex Stationary White Noise Wave
    ```
    """)

    st.markdown("---")
    st.markdown("### 🔬 Interactive Pipeline Stage Inspector")

    # Load sample image for demonstration
    sample_path = os.path.join("data", "sample_input.png")
    if os.path.exists(sample_path):
        sample_img = np.asarray(Image.open(sample_path).convert("L"), dtype=np.float64) / 255.0
    else:
        # Generate synthetic target pattern
        x, y = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
        sample_img = np.where((x**2 + y**2) < 0.5, 0.9, 0.1)

    P1_sim = generate_phase_mask(sample_img.shape)
    P2_sim = generate_phase_mask(sample_img.shape)
    stages = generate_drpe_intermediate_stages(sample_img, P1_sim, P2_sim)

    stage_option = st.selectbox(
        "Select Pipeline Stage to Inspect:",
        [
            "Stage 1: Original Spatial Image I(x, y)",
            "Stage 2: Spatial Phase Modulation (I · P1)",
            "Stage 3: 2D Fourier Spectrum F{ I · P1 }",
            "Stage 4: Spectral Phase Modulation (F · P2)",
            "Stage 5: Final Ciphertext C(x, y) (Complex White Noise)",
            "Stage 6: Decryption Stage & Failure Mode (Correct vs Invalid Key)"
        ]
    )

    sc1, sc2 = st.columns([1, 1], gap="medium")

    if stage_option.startswith("Stage 1"):
        with sc1:
            st.image(stages["stage1_input"], caption="Original Image I(x, y)", width='stretch', clamp=True)
        with sc2:
            st.markdown(r"""
            #### Stage 1: Spatial Domain Input
            - **Function**: Represents the raw image pixel intensity $I(x, y) \in [0, 1]$.
            - **Mathematical Domain**: Spatial Coordinates $(x, y)$.
            - **Property**: Contains structured visual features, object boundaries, and spatial correlations.
            """)

    elif stage_option.startswith("Stage 2"):
        with sc1:
            st.image(stages["stage2_mag"], caption="Magnitude |I(x, y) · P1(x, y)|", width='stretch', clamp=True)
        with sc2:
            st.image(stages["stage2_phase"], caption="Phase Angle ∠(I · P1) [0, 2π]", width='stretch', clamp=True)
            st.markdown(r"""
            #### Stage 2: Spatial Phase Modulation
            - **Formula**: $f(x, y) = I(x, y) \cdot P_1(x, y) = I(x, y) \cdot e^{j 2\pi r_1(x, y)}$
            - **Function**: Multiplies input pixels by a uniform random phase mask $P_1(x, y)$ on the complex unit circle.
            - **Effect**: Scrambles the spatial phase while maintaining the original magnitude $|I(x, y)|$.
            """)

    elif stage_option.startswith("Stage 3"):
        with sc1:
            st.image(stages["stage3_mag_log"], caption="Log Magnitude Spectrum log(1 + |FFT|)", width='stretch', clamp=True)
        with sc2:
            st.image(stages["stage3_phase"], caption="Fourier Phase Spectrum ∠FFT", width='stretch', clamp=True)
            st.markdown(r"""
            #### Stage 3: 2D Fourier Transformation
            - **Formula**: $F(u, v) = \mathcal{F}\left\{ I(x, y) \cdot P_1(x, y) \right\}$
            - **Function**: Transforms spatial signals into spatial frequency coordinates $(u, v)$ using 2D FFT.
            - **Effect**: Spreads spatial information evenly across the entire frequency spectrum.
            """)

    elif stage_option.startswith("Stage 4"):
        with sc1:
            st.image(stages["stage4_mag_log"], caption="Modulated Log Spectrum log(1 + |F · P2|)", width='stretch', clamp=True)
        with sc2:
            st.image(stages["stage4_phase"], caption="Modulated Phase Spectrum ∠(F · P2)", width='stretch', clamp=True)
            st.markdown(r"""
            #### Stage 4: Fourier Plane Spectral Phase Modulation
            - **Formula**: $G(u, v) = F(u, v) \cdot P_2(u, v) = F(u, v) \cdot e^{j 2\pi r_2(u, v)}$
            - **Function**: Multiplies the frequency spectrum by independent random phase mask $P_2(u, v)$.
            - **Effect**: Completely randomizes the spectral phase components required to reconstruct the image.
            """)

    elif stage_option.startswith("Stage 5"):
        with sc1:
            st.image(stages["stage5_cipher_mag"], caption="Ciphertext Magnitude |C(x, y)|", width='stretch', clamp=True)
        with sc2:
            st.image(stages["stage5_cipher_phase"], caption="Ciphertext Phase ∠C(x, y)", width='stretch', clamp=True)
            st.markdown(r"""
            #### Stage 5: Final Ciphertext (Stationary White Noise)
            - **Formula**: $C(x, y) = \mathcal{F}^{-1}\left\{ \mathcal{F}\left\{ I \cdot P_1 \right\} \cdot P_2 \right\}$
            - **Property**: Complex-valued wave function whose amplitude and phase form stationary white Gaussian noise.
            - **Security**: Without knowledge of both phase masks ($P_1, P_2$), no statistical or visual information can be extracted.
            """)

    elif stage_option.startswith("Stage 6"):
        with sc1:
            st.image(stages["stage6_correct"], caption="Decryption with Correct Key (Exact Recovery)", width='stretch', clamp=True)
        with sc2:
            st.image(stages["stage6_wrong"], caption="Decryption with Invalid Key (Zero Visual Info)", width='stretch', clamp=True)
            st.markdown(r"""
            #### Stage 6: Decryption & Key Sensitivity
            - **Decryption Formula**:
              $$\text{Decrypted} = \left| \mathcal{F}^{-1}\left\lbrace \mathcal{F}\lbrace C(x, y) \rbrace \cdot P_2^{\ast}(u, v) \right\rbrace \cdot P_1^{\ast}(x, y) \right|$$
            - **Exact Recovery**: Multiplying by complex conjugates $P_2^*$ and $P_1^*$ cancels out both phase delays, yielding the original image with zero loss ($10^{-15}$ precision).
            - **Wrong Key Failure**: Using even a slightly incorrect key leaves random phase residual noise, producing pure stationary noise.
            """)
