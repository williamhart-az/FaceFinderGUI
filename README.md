# FaceFinder GUI: Local AI Photo Organizer

**FaceFinder GUI** is a powerful, privacy-focused desktop application that uses state-of-the-art Facial Recognition (DeepFace) to scan massive local photo archives. It identifies specific people based on reference photos and organizes the matches into a dedicated output folder.

Unlike cloud services, this runs entirely on your local machine. It is designed for "data hoarders," photographers, and archivists with large, unorganized libraries.

## 🚀 Key Features

* **State-of-the-Art AI:** Uses **ArcFace** (Model) and **RetinaFace** (Detector) by default for industry-leading accuracy, even on side profiles or blurry images.
* **Incremental Indexing:** Scans are persistent. If you stop the analysis, it resumes where it left off. It saves face embeddings to `.pkl` files, so you only need to process an image once.
* **"Pure Math" Search Mode:** Once your archive is indexed, you can toggle "Skip Indexing" to search through tens of thousands of photos in seconds using vector math.
* **Self-Healing Database:** Automatically detects and prunes missing or corrupt files from the index without crashing.
* **Smart Image Handling:** Automatically handles EXIF rotation (orientation) and converts PNG Alpha channels to ensure accurate detection.
* **Safety Backups:** Automatically creates backups of your index files before modification to prevent data loss.
* **Interactive GUI:**
    * **Tooltips:** Hover over any setting for an explanation.
    * **Min Distance Calculator:** Test your reference photos against the database to find the perfect threshold before running a full copy.
    * **Direct File Access:** Clickable links in the results log to open matched images instantly.

## 🛠️ Installation

### Prerequisites
* Python 3.8+
* A CUDA-capable GPU is highly recommended (but not required) for speed.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/FaceFinder-GUI.git](https://github.com/YOUR_USERNAME/FaceFinder-GUI.git)
cd FaceFinder-GUI
```

### 2. Install Dependencies
```bash
pip install deepface pandas numpy pillow tf-keras
```
*(Note: `tkinter` is usually included with Python standard installations)*

### 3. Run the Application
```bash
python facefinder_gui.py
```

## 📖 Usage Guide

1.  **Configure Archive:** Click "Add" under **Archive Directories** to select the folders you want to scan (e.g., `D:/Photos/2020`, `E:/Backups`).
2.  **Set Output:** Choose an **Output Directory**. This is where matched photos will be copied (they are *copied*, not moved, so your original archive remains untouched).
3.  **Add References:** Click **Add Person/Refs**. Enter a name (e.g., "Ono") and select 1-3 clear photos of that person.
    * *Tip: Use the "Calculate Min Distances" button to test which reference photo gets the best score (lowest number).*
4.  **Tune Settings (Optional):**
    * **Model:** `ArcFace` is recommended.
    * **Detector:** `RetinaFace` is most accurate; `MediaPipe` is faster.
    * **Max Distance:** The threshold for a match. For ArcFace/Cosine, **0.40** is a good starting point. Lower is stricter; higher includes more false positives.
5.  **Run Analysis:**
    * **First Run:** Leave "Skip Indexing" **unchecked**. The app will scan every face and build the database.
    * **Subsequent Runs:** Check **"Skip Indexing"**. The app will load the existing database and perform the search instantly.

# 🚀 How to Run FaceFinder with `uv`

This project is optimized for [uv](https://github.com/astral-sh/uv), an extremely fast Python package manager. It handles Python versions and dependencies automatically.

## 1. Install uv (Windows)

Open **PowerShell** (Start Menu -> type "PowerShell") and paste this command:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

*Tip: You may need to close and reopen PowerShell after installing for the 'uv' command to be recognized.*

# FaceFinderGUI[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]

A Python-based GUI application for face detection and recognition, powered by TensorFlow.[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]

## Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (for dependency management)[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]
- Windows (for the specific GPU setup instructions below)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/williamhart-az/FaceFinderGUI.git
   cd FaceFinderGUI
   ```[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]

2. **Install dependencies**
   Use `uv` to set up your environment and install the required packages:
   ```bash
   uv sync
   ```[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]

---

## Windows GPU Setup (The "DLL Drop")[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]

To enable GPU acceleration with TensorFlow on Windows without a full system-wide CUDA installation, follow these specific steps to "drop" the required DLLs directly into your script folder.

### 1. Get the CUDA 11.2 DLLs
1. Go to the **[NVIDIA CUDA Toolkit 11.2.2 Archive](https://developer.nvidia.com/cuda-11.2.2-download-archive)**.[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]
2. **Download**: Select **Windows** -> **x86_64** -> **10** (or 11) -> **exe (local)**.[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]
3. **Extract**: You don't actually have to install it.[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)] You can open the `.exe` file with **7-Zip** (Right-click -> 7-Zip -> Open archive).[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)] Alternatively, you can install it, copy the files, and then uninstall.[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]
4. **Find & Copy**: Locate the following files inside the `bin/` folder of the archive and copy them to your script folder (the same folder where `faceFindGUI_0.1.py` is located):

   - `cudart64_110.dll`[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]
   - `cublas64_11.dll`[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]
   - `cublasLt64_11.dll`[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]
   - `cufft64_10.dll`[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]
   - `cusparse64_11.dll`[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]

### 2. Get the cuDNN 8.1 DLLs
TensorFlow needs the Deep Neural Network library (cuDNN) to match CUDA 11.2.[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]

1. Go to the **[NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)**.[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)][[2](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQG7rtfyOHhoQkxQI4wRWmXu8hyP7CGazlDchAqEdejsFheKOSzAOS9084pGQjZhMsRNuJfjQCn4JD543Uy71r2ItARN6pxZ7U2z_rb5AXVOt2UvtQbxoyo45JLiJo5MIt_y7L47eQ5hET3bzxhnnaZc_kZ9zL4iaEwnmq7f8tqfJ4Y3vdFxexIX6gqf9EaWxDIqRcf94nJOXRmeCUj1cd52ig%3D%3D)][[3](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQEc3i-MmXHD28jkZIaenZPMNGT10_8KYds3Pmd4mpx5060rg3xVnn0fhGKn83cVfii4ux8l2I0MoDf8LtlSFm1atDlxdVYIoPZe1xk9sZywh7Nsev5JupofqJiGsXSmw7qCSydqoFA%3D)]
2. **Download**: Select **"Download cuDNN v8.1.1 (Feb 26th, 2021), for CUDA 11.0, 11.1 and 11.2"**.[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]
   *(Note: You will need an NVIDIA developer account to download this, or you can search for "cudnn 8.1 windows zip" on a trusted mirror).*[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]
3. **Extract**: Unzip the downloaded file.
4. **Find & Copy**: Locate the following file inside the `bin/` folder and copy it to your script folder:

   - `cudnn64_8.dll`[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]

### 3. Verification
Once those ~6 DLL files are sitting in your folder (e.g., `C:\Users\YourName\Downloads\`) next to `faceFindGUI_0.1.py`:

1. Run the `uv` command again to start the application.[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]
2. TensorFlow will check the current directory first, find the DLLs, and initialize the GPU.[[1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fvertexaisearch.cloud.google.com%2Fgrounding-api-redirect%2FAUZIYQFw-X6w7hd81YaClRSeHugONpyTP36pQwuBsAxo3-O8y3gw6UhSkeb00Dn-qH_lJ5fjyPhQLFNEFzUdnUcNhpj_EV3H4ARunebuOa1HxopVbKNJfZ9h6TLWAsLLOrpJJZGYqgU9rW5n-D_T2RjDxnKc3-RRNfND9lIjZtxWSp6gf8wcURYhdtstr28d4P-NyBF7EHcF4J-QNgz0hEMrjT6tVbrbDanPNGBmrZlx1PRQRofwFI2OgL0K)]

You should see the following success message in your console:

> ✅ **GPU DETECTED**

---

## Usage

Run the application using `uv`:

```bash
uv run faceFindGUI_0.1.py
## 2. Run the Application (The Easy Way)

You do not need to manually create virtual environments or install pip packages. `uv` does it all in one step.

Open your terminal in this folder and run:

```bash
uv run facefinder_gui.py
```

**What happens next?**
1. `uv` checks the script for required libraries (`deepface`, `pandas`, etc.).
2. It creates a temporary, isolated environment.
3. It installs the libraries instantly.
4. It launches the GUI.

---

## Alternative: Set up a Permanent Environment

If you prefer to "install" the project permanently in this folder:

1. **Initialize the project:**
   ```bash
   uv init
   ```

2. **Add the dependencies (runs once):**
   ```bash
   uv add deepface pandas numpy pillow tf-keras
   ```

3. **Run the script:**
   ```bash
   uv run facefinder_gui.py
   ```


## ⚙️ Configuration Details

* **ArcFace + Cosine:** The "Gold Standard" for recognition.
* **Euclidean L2:** An alternative distance metric (requires different threshold values).
* **RetinaFace:** Excellent at detecting faces in crowds, at angles, or partially obscured.

## 🛡️ Privacy & Local Processing
This application relies on the `deepface` library. **No images are uploaded to the cloud.** All facial recognition and processing happen locally on your CPU/GPU. Your biometric data remains on your hard drive.

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License
[MIT](https://choosealicense.com/licenses/mit/)
