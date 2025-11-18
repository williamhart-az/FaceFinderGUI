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
