# 🔐 Fourier-Domain Image Encryptor

An optical cryptography suite implementing **Double Random Phase Encoding (DRPE)** in the Fourier domain using a 4-$f$ optical correlator architecture. Supports grayscale and full-colour RGB image encryption, password-protected key library storage (AES-256), quantitative reconstruction metrics (PSNR, MSE), pixel intensity distribution histograms, real-time brute-force key match scoring, and a modern glassmorphic Streamlit web application.

---

## 🌟 Key Features

- **Double Random Phase Encoding (DRPE)**:
  - 2D Fast Fourier Transform (FFT)-based optical image encryption.
  - Spatial domain mask ($P_1$) and Fourier plane spectral mask ($P_2$).
  - Theoretical near-zero reconstruction error ($\sim 10^{-15}$ float precision).
  - Discrete 8-bit rounding exact **SHA-256** hash match verification.
- **RGB & Grayscale Processing**:
  - Independent channel-wise encryption and decryption for full-colour RGB images.
  - Automatic mode detection on upload (grayscale vs. RGB) with per-channel DRPE applied uniformly.
- **Secure Key Management**:
  - Unique human-readable key identifiers (`KEY-XXXX-YYYY`).
  - Password-protected `.ekey` bundles using **AES-256 (Fernet)** with **PBKDF2-HMAC-SHA256** (480,000 iterations).
- **Cryptographic Analytics & Quality Metrics**:
  - **Pixel Intensity Histograms**: Compare original pixel distribution vs. ciphertext magnitude (stationary white noise) vs. decrypted image.
  - **Brute-Force Key Match Score Plot**: Live bar chart rendered during decryption showing the exact image smoothness gradient score for each tested key.
  - **Phase Mask Inspector**: 2D spatial phase maps and 1D angle distribution histograms $[0, 2\pi]$ to verify uniform randomness.
  - **Adjacent Pixel Correlation Analysis**: Scatter plots of neighboring pixel pairs (horizontal, vertical, or diagonal) contrasting the strong correlation in real images against the near-zero correlation of the ciphertext.
- **Interactive Visual Pipeline Inspector**:
  - Step-by-step visual walkthrough of all 6 mathematical stages of the DRPE 4-$f$ optical correlator system.
- **Dual Interfaces**:
  - **Streamlit Web Application**: Modern dark-mode glassmorphic dashboard.
  - **Python CLI**: Standalone pipeline script for quick testing and automated batch workflows.

---

## 📐 Mathematical Principle: DRPE

Double Random Phase Encoding simulates a 4-$f$ optical correlator system:

```text
Input Image I(x, y)
       │
       ▼
 [ × P1(x, y) ]  ──> Spatial phase modulation: P1 = exp(j * 2π * r1)
       │
       ▼
  [ 2D FFT ]     ──> Fourier spectrum transformation
       │
       ▼
 [ × P2(u, v) ]  ──> Fourier plane phase modulation: P2 = exp(j * 2π * r2)
       │
       ▼
  [ 2D IFFT ]    ──> Inverse Fourier transform
       │
       ▼
Ciphertext C(x, y) ──> Stationary white Gaussian noise complex wave
```

### Decryption:

$$
\text{Decrypted Image} = \left| \mathcal{F}^{-1}\left\lbrace \mathcal{F}\lbrace C(x, y) \rbrace \cdot P_2^{\ast}(u, v) \right\rbrace \cdot P_1^{\ast}(x, y) \right|
$$

Where $P_1^{\ast}$ and $P_2^{\ast}$ denote the complex conjugates of phase masks $P_1$ and $P_2$. Without both correct phase masks, decryption yields zero visual information and appears as random noise.

---

## 📁 Project Structure

```text
Fourier-Domain-Image-Encryptor/
├── app.py                  # Streamlit web application (Glassmorphic UI & 3 Tabs)
├── main.py                 # Standalone CLI demo script
├── cmd.txt                 # Quick command reference
├── requirements.txt        # Python package dependencies
├── core/
│   ├── __init__.py         # Core package exports
│   ├── drpe.py             # DRPE encryption/decryption engine (Grayscale & RGB)
│   ├── image_io.py         # Image loading, saving, and canvas padding helpers
│   ├── key_library.py      # Key pair generation & AES-256 encrypted storage
│   └── metrics.py          # PSNR, MSE, image smoothness score, SHA-256 hash
├── utils/
│   └── visualizations.py   # Pixel histograms, phase spectra, key score bar charts
├── tests/
│   ├── test_drpe.py        # Unit tests for DRPE mathematics & hash matching
│   └── test_key_library.py # Unit tests for key storage & serialization
├── data/                   # Input sample images
└── outputs/                # CLI generated ciphertexts and decrypted images
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/ahs1nat/Fourier-Domain-Image-Encryptor.git
cd Fourier-Domain-Image-Encryptor

# Create virtual environment
python -m venv .venv

# Activate virtual environment:
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Run the Streamlit Web Application

Launch the interactive web UI:

```bash
streamlit run app.py
```

*or via virtual environment executable:*
```bash
.venv\Scripts\streamlit.exe run app.py
```

Then open your browser at `http://localhost:8501`.

#### Available Web Tabs:
1. **🔒 Encrypt**: Upload an image, generate a 10-key phase mask library, set a password for the `.ekey` bundle, download the ciphertext (`.npy`) and key library, and inspect a 2-panel pixel distribution histogram (original vs. ciphertext).
2. **🔓 Decrypt**: Upload `.ekey` and `.npy`, unlock with a password, brute-force the key library against the ciphertext with live per-key scoring and a SHA-256 hash match check, browse every key attempt in a thumbnail grid, and view a bar chart of image-smoothness scores across all tested keys.
3. **📘 How DRPE Works**: Step through an interactive 6-stage visual pipeline (with PSNR/MSE metric cards on the decryption stage), plus supporting analysis panels for Phase Mask Inspection ($P_1$/$P_2$ spatial and spectral phase maps), a 3-panel pixel distribution histogram (original vs. ciphertext vs. decrypted), and Adjacent Pixel Correlation Analysis.

---

### 3. Run the CLI Script

Run the command-line demo pipeline:

```bash
python main.py
```

**What it does:**
- Automatically creates a synthetic test pattern (`data/sample_input.png`) if none exists.
- Generates random phase masks ($P_1, P_2$).
- Encrypts the image to `outputs/encrypted.png`.
- Decrypts the ciphertext to `outputs/decrypted.png`.
- Computes and prints the reconstruction error.

---

### 4. Running Unit Tests

Execute the automated test suite:

```bash
python -m unittest discover -s tests -v
```

---

## 🔒 Security Specifications

- **Phase Masks**: Uniformly distributed independent random variables $r_1, r_2 \sim \mathcal{U}[0, 1)$ over the complex unit circle $e^{j 2\pi r}$.
- **Key File Encryption (`.ekey`)**:
  - **Algorithm**: AES-256 in CBC mode with HMAC (Fernet).
  - **Key Derivation Function**: PBKDF2-HMAC-SHA256.
  - **Work Factor**: 480,000 iterations.
  - **Salt**: 32-byte cryptographically secure random salt (`os.urandom(32)`).
