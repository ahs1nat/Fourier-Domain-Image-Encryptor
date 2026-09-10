# 🔐 Fourier-Domain Image Encryptor

An optical cryptography suite implementing **Double Random Phase Encoding (DRPE)** in the Fourier domain. Supports grayscale and full-colour RGB image encryption, password-protected key storage (AES-256), quantitative reconstruction metrics (PSNR, MSE, SSIM, Entropy), noise robustness testing, and a modern Streamlit web interface.

---

## 🌟 Key Features

- **Double Random Phase Encoding (DRPE)**:
  - 2D Fast Fourier Transform (FFT)-based optical image encryption.
  - Spatial domain mask ($P_1$) and Fourier domain spectral mask ($P_2$).
  - Theoretical near-zero reconstruction error ($\sim 10^{-15}$ float precision).
- **RGB & Grayscale Processing**:
  - Independent channel-wise encryption and decryption for full-colour RGB images.
  - Automatic padding to standardized canvas sizes with exact crop-back on recovery.
- **Secure Key Management**:
  - Unique human-readable key identifiers (`KEY-XXXX-YYYY`).
  - Standard `.pkl` key bundles and AES-256 password-protected `.ekey` files using **PBKDF2-HMAC-SHA256** (480,000 iterations).
- **Cryptographic & Quality Metrics**:
  - Peak Signal-to-Noise Ratio (**PSNR**), Mean Squared Error (**MSE**), and Structural Similarity Index (**SSIM**).
  - Shannon Entropy calculation for ciphertext randomness verification.
- **Transmission & Noise Robustness Simulator**:
  - Simulate real-world transmission channel noise: **Gaussian noise** and **Salt-and-Pepper noise**.
  - Evaluate image recovery fidelity under varied noise intensities.
- **Dual Interfaces**:
  - **Streamlit Web Application**: Modern dark-mode glassmorphic dashboard with live phase mask inspection.
  - **Python CLI**: Standalone pipeline for automated batch workflows and quick testing.

---

## 📐 Mathematical Principle: DRPE

Double Random Phase Encoding simulates a 4-$f$ optical correlator:

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
$$\text{Decrypted Image} = \left| \mathcal{F}^{-1}\left\{ \mathcal{F}\{C(x, y)\} \cdot P_2^*(u, v) \right\} \cdot P_1^*(x, y) \right|$$

Where $P_1^*$ and $P_2^*$ denote the complex conjugates of phase masks $P_1$ and $P_2$. Without both correct masks, decryption yields zero visual information and appears as random noise.

---

## 📁 Project Structure

```text
Fourier-Domain-Image-Encryptor/
├── app.py                  # Streamlit web application
├── main.py                 # Standalone CLI demo script
├── cmd.txt                 # Quick command reference
├── requirements.txt        # Python package dependencies
├── core/
│   ├── __init__.py         # Core package exports
│   ├── drpe.py             # DRPE encryption/decryption (Grayscale & RGB)
│   ├── image_io.py         # Image loading, saving, and canvas padding
│   ├── key_library.py      # Key generation, management & AES-256 encryption
│   └── metrics.py          # PSNR, SSIM, MSE, Entropy & noise robustness
├── tests/
│   ├── test_drpe.py        # Unit tests for DRPE mathematics & edge cases
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

Launch the interactive UI:

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

#### Available Tabs:
1. **Encrypt**: Upload an image (PNG, JPG), choose grayscale or RGB, generate or select key pairs, optionally encrypt the key with a password (`.ekey`), and download ciphertext (`.npy`) and keys.
2. **Decrypt**: Upload ciphertext and corresponding key file, decrypt image, and view PSNR/SSIM/MSE metrics. Includes the **Noise Robustness Tester** expander.
3. **Phase Mask Inspector**: Visualize real, imaginary, and phase angle components of generated random phase masks.

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

- **Phase Masks**: Uniformly distributed independent random variables $r_1, r_2 \sim \mathcal{U}[0, 1)$ over the complex unit circle.
- **Key File Encryption (`.ekey`)**:
  - **Algorithm**: AES-256 in CBC mode with HMAC (Fernet).
  - **Key Derivation Function**: PBKDF2-HMAC-SHA256.
  - **Work Factor**: 480,000 iterations.
  - **Salt**: 32-byte cryptographically secure random salt (`os.urandom(32)`).

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.