# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "deepface",
#     "pandas",
#     "numpy<2",
#     "pillow",
#     "tf-keras",
#     # --- Platform Specifics ---
#     "tensorflow==2.10.1 ; sys_platform == 'win32'",
#     "tensorflow[and-cuda] ; sys_platform == 'linux'",
#     "tensorflow-macos ; sys_platform == 'darwin'",
#     "tensorflow-metal ; sys_platform == 'darwin'",
# ]
# ///
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog
import os
import shutil
import pickle
import time
import pandas as pd
import subprocess
import platform
import webbrowser
from deepface import DeepFace
from PIL import Image, ImageTk, ImageOps
import threading
import numpy as np
import tensorflow as tf

# --- Configuration ---
ARCHIVE_DIRS = []
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "DeepFaceHits")
REFERENCE_IMAGES_CONFIG = {}

MODEL = "ArcFace"
DETECTOR = "retinaface"
DIST_METRIC = "cosine"
MAX_DIST = 0.28

# New Configuration Lists
EXCLUDED_FOLDER_NAMES = ["$RECYCLE.BIN", "System Volume Information", ".git", "__pycache__"]
ENABLED_EXTENSIONS = {
    ".jpg": True, ".jpeg": True, ".png": True, 
    ".bmp": True, ".gif": True, ".webp": True, ".tiff": True
}

CHECKPOINT_INTERVAL_SECONDS = 5 * 60  # 5 minutes

# ========== GPU CHECK HELPER ==========
def check_gpu_status():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return f"✅ GPU DETECTED: {len(gpus)} device(s) active.\n   DeepFace will use hardware acceleration."
        else:
            return "⚠️ NO GPU DETECTED. DeepFace is running on CPU.\n   Analysis will be significantly slower."
    except Exception as e:
        return f"⚠️ Error checking GPU status: {str(e)}"

# ========== TOOLTIP CLASS (FIXED) ==========
class ToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(500, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        try:
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        except:
            return

        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

# ========== GUI APPLICATION CLASS ==========
class FaceFinderGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DeepFace GUI Face Finder v0.4.3 (Volume Fix)")
        self.geometry("1200x950") 

        self.hits_log_df = pd.DataFrame()
        self.running_thread = None
        self.stop_event = threading.Event() 
        
        self.skip_indexing_var = tk.BooleanVar(value=True) 
        
        # Extension Vars for Checkboxes
        self.ext_vars = {}
        for ext in ENABLED_EXTENSIONS.keys():
            self.ext_vars[ext] = tk.BooleanVar(value=True)

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.HITS_LOG = os.path.join(OUTPUT_DIR, "hits_log.csv")

        self._create_menu()
        self._create_widgets()
        self._load_initial_config() 

        self.write("--- SYSTEM CHECK ---\n")
        self.write(check_gpu_status() + "\n")
        self.write("-" * 20 + "\n\n")

    def _create_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Manual / Model Guide", command=self._show_manual)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "FaceFinder GUI v0.4.3\nPowered by DeepFace"))

    def _add_tooltip(self, widget, text):
        ToolTip(widget, text)

    def _create_widgets(self):
        # === MAIN CONFIGURATION FRAME ===
        config_frame = tk.LabelFrame(self, text="Configuration", padx=10, pady=5)
        config_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # 1. Archive Dirs
        tk.Label(config_frame, text="Archive Directories:").grid(row=0, column=0, sticky="w", pady=2)
        self.archive_dirs_listbox = tk.Listbox(config_frame, height=3, width=70)
        self.archive_dirs_listbox.grid(row=0, column=1, columnspan=2, pady=2)
        self._add_tooltip(self.archive_dirs_listbox, "The folders containing the massive photo collections you want to search through.")
        
        btn_add_arc = tk.Button(config_frame, text="Add", command=self._add_archive_dir)
        btn_add_arc.grid(row=0, column=3, padx=2)
        tk.Button(config_frame, text="Remove", command=self._remove_archive_dir).grid(row=0, column=4, padx=2)

        # 2. Output Dir
        tk.Label(config_frame, text="Output Directory:").grid(row=1, column=0, sticky="w", pady=2)
        self.output_dir_entry = tk.Entry(config_frame, width=70)
        self.output_dir_entry.insert(0, OUTPUT_DIR) 
        self.output_dir_entry.grid(row=1, column=1, columnspan=2, pady=2)
        
        tk.Button(config_frame, text="Browse", command=self._select_output_dir).grid(row=1, column=3, padx=2)

        # 3. Ref Images
        tk.Label(config_frame, text="Reference Images (Person: Paths):").grid(row=2, column=0, sticky="w", pady=2)
        self.ref_images_listbox = tk.Listbox(config_frame, height=4, width=70)
        self.ref_images_listbox.grid(row=2, column=1, columnspan=2, pady=2)
        
        tk.Button(config_frame, text="Add Person/Refs", command=self._add_reference_person).grid(row=2, column=3, padx=2)
        tk.Button(config_frame, text="Remove Person", command=self._remove_reference_person).grid(row=2, column=4, padx=2)

        # 4. Model Settings
        tk.Label(config_frame, text="Model:").grid(row=3, column=0, sticky="w", pady=2)
        self.model_var = tk.StringVar(self)
        self.model_var.set(MODEL)
        models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
        om_model = tk.OptionMenu(config_frame, self.model_var, *models)
        om_model.grid(row=3, column=1, sticky="ew", pady=2)

        tk.Label(config_frame, text="Detector:").grid(row=3, column=2, sticky="w", pady=2)
        self.detector_var = tk.StringVar(self)
        self.detector_var.set(DETECTOR)
        detectors = ["opencv", "
