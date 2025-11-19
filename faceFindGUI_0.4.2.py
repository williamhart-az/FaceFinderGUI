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
        # FIX: Use absolute widget coordinates instead of 'bbox("insert")'
        # This prevents crashes on Listboxes and Buttons.
        try:
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        except:
            # Fallback if widget is somehow not ready
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
        self.title("DeepFace GUI Face Finder v0.4.3")
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
        detectors = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe", "yolov8"]
        om_detect = tk.OptionMenu(config_frame, self.detector_var, *detectors)
        om_detect.grid(row=3, column=3, sticky="ew", pady=2)

        tk.Label(config_frame, text="Distance Metric:").grid(row=4, column=0, sticky="w", pady=2)
        self.metric_var = tk.StringVar(self)
        self.metric_var.set(DIST_METRIC)
        metrics = ["cosine", "euclidean", "euclidean_l2"]
        om_metric = tk.OptionMenu(config_frame, self.metric_var, *metrics)
        om_metric.grid(row=4, column=1, sticky="ew", pady=2)

        tk.Label(config_frame, text="Max Distance:").grid(row=4, column=2, sticky="w", pady=2)
        self.max_dist_entry = tk.Entry(config_frame, width=10)
        self.max_dist_entry.insert(0, str(MAX_DIST))
        self.max_dist_entry.grid(row=4, column=3, sticky="w", pady=2)

        # === FILTERS & EXCLUSIONS FRAME ===
        filter_frame = tk.LabelFrame(self, text="Filters & Exclusions", padx=10, pady=5)
        filter_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # File Type Checkboxes
        tk.Label(filter_frame, text="Scan Filetypes:").grid(row=0, column=0, sticky="nw", pady=2)
        checkbox_frame = tk.Frame(filter_frame)
        checkbox_frame.grid(row=0, column=1, sticky="w")
        
        col = 0
        for ext, var in self.ext_vars.items():
            cb = tk.Checkbutton(checkbox_frame, text=ext, variable=var)
            cb.grid(row=0, column=col, padx=2)
            col += 1

        # Directory Exclusions
        tk.Label(filter_frame, text="Exclude Folders (Name):").grid(row=1, column=0, sticky="nw", pady=2)
        self.exclude_listbox = tk.Listbox(filter_frame, height=3, width=60)
        self.exclude_listbox.grid(row=1, column=1, sticky="w", pady=2)
        self._add_tooltip(self.exclude_listbox, "Folder names to completely ignore (e.g., 'thumbs', '$RECYCLE.BIN').")

        btn_exc_frame = tk.Frame(filter_frame)
        btn_exc_frame.grid(row=1, column=2, sticky="nw", padx=5)
        tk.Button(btn_exc_frame, text="Add Name", command=self._add_exclude_name).pack(fill=tk.X, pady=1)
        tk.Button(btn_exc_frame, text="Remove", command=self._remove_exclude_name).pack(fill=tk.X, pady=1)

        # === ACTIONS & LOGS ===
        action_results_frame = tk.Frame(self, padx=10, pady=5)
        action_results_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        cb_skip = tk.Checkbutton(
            action_results_frame, 
            text="Skip Indexing (Use existing PKL - Pure Math Mode)", 
            variable=self.skip_indexing_var
        )
        cb_skip.pack(anchor="w", pady=(0, 5))

        button_frame = tk.Frame(action_results_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        btn_run = tk.Button(button_frame, text="Run Analysis", command=self._start_analysis, bg="green", fg="white")
        btn_run.pack(side=tk.LEFT, padx=5)

        btn_index = tk.Button(button_frame, text="Build/Update Indexes Only", command=self._start_indexing_only, bg="#2980b9", fg="white")
        btn_index.pack(side=tk.LEFT, padx=5)

        btn_stop = tk.Button(button_frame, text="Stop Analysis", command=self._stop_analysis, bg="red", fg="white")
        btn_stop.pack(side=tk.LEFT, padx=5)
        
        btn_calc = tk.Button(button_frame, text="Calculate Min Distances", command=self._calculate_min_distances_optimized)
        btn_calc.pack(side=tk.LEFT, padx=5)

        tk.Button(button_frame, text="Save Configuration", command=self._save_config).pack(side=tk.LEFT, padx=5)

        tk.Label(action_results_frame, text="Min Distances for Reference Images:").pack(anchor="w", pady=5)
        self.min_dist_display = scrolledtext.ScrolledText(action_results_frame, height=8, wrap=tk.WORD)
        self.min_dist_display.pack(fill=tk.X, pady=2)

        tk.Label(action_results_frame, text="Console Output:").pack(anchor="w", pady=5)
        self.console_output = scrolledtext.ScrolledText(action_results_frame, height=10, wrap=tk.WORD)
        self.console_output.pack(fill=tk.BOTH, expand=True)

        tk.Label(action_results_frame, text="Reference Image Previews:").pack(anchor="w", pady=5)
        self.image_preview_frame = tk.Frame(action_results_frame)
        self.image_preview_frame.pack(fill=tk.X, pady=5)

        self.ref_images_listbox.bind("<<ListboxSelect>>", self._show_selected_ref_images)

        self.console_output.tag_config('info', foreground='black')
        self.console_output.tag_config('warning', foreground='orange')
        self.console_output.tag_config('error', foreground='red')
        self.console_output.tag_config('success', foreground='green')

        import sys
        sys.stdout = self
        sys.stderr = self

        status_frame = tk.Frame(self, bd=1, relief=tk.SUNKEN)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.timer_var = tk.StringVar()
        self.timer_var.set("")
        self.timer_label = tk.Label(status_frame, textvariable=self.timer_var, width=20, anchor=tk.E, font=("Arial", 9, "bold"), fg="blue")
        self.timer_label.pack(side=tk.RIGHT, padx=5)

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(status_frame, textvariable=self.status_var, anchor=tk.W, font=("Arial", 9))
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _update_status(self, message):
        self.after(0, lambda: self.status_var.set(message))

    def _update_timer(self, message):
        self.after(0, lambda: self.timer_var.set(message))

    def _show_manual(self):
        manual_win = tk.Toplevel(self)
        manual_win.title("User Manual & Guide")
        manual_win.geometry("800x600")
        txt = scrolledtext.ScrolledText(manual_win, wrap=tk.WORD, padx=10, pady=10)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.tag_config("h1", font=("Segoe UI", 14, "bold"), foreground="#2c3e50")
        txt.tag_config("h2", font=("Segoe UI", 12, "bold"), foreground="#2980b9")
        txt.insert(tk.END, "FaceFinder GUI Manual\n\n", "h1")
        txt.insert(tk.END, "Real-Time Matching enabled. Matches are copied immediately during indexing.\n")
        txt.configure(state=tk.DISABLED)

    def write(self, message):
        self.console_output.insert(tk.END, message)
        if "⚠️" in message or "warning" in message.lower():
            self.console_output.tag_add('warning', self.console_output.index('end-1c linestart'), self.console_output.index('end-1c'))
        elif "Error" in message or "Traceback" in message or "fail" in message.lower():
            self.console_output.tag_add('error', self.console_output.index('end-1c linestart'), self.console_output.index('end-1c'))
        elif "✅" in message or "success" in message.lower():
            self.console_output.tag_add('success', self.console_output.index('end-1c linestart'), self.console_output.index('end-1c'))
        else:
            self.console_output.tag_add('info', self.console_output.index('end-1c linestart'), self.console_output.index('end-1c'))
        self.console_output.see(tk.END)

    def flush(self): pass

    def _load_image_fixed(self, path):
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB") 
            img_np = np.array(img)
            img_np = img_np[:, :, ::-1] 
            return img_np
        except Exception as e:
            return path 

    def _open_file_from_link(self, filepath):
        print(f"🖱️ Link clicked! Attempting to open: {filepath}")
        if not os.path.exists(filepath):
            messagebox.showerror("File Not Found", f"Could not find file at:\n{filepath}")
            return
        try:
            if platform.system() == 'Windows': os.startfile(filepath)
            elif platform.system() == 'Darwin': subprocess.call(('open', filepath))
            else: subprocess.call(('xdg-open', filepath))
        except: pass

    # --- Listbox Helpers ---
    def _add_archive_dir(self):
        directory = filedialog.askdirectory()
        if directory and directory not in self.archive_dirs_listbox.get(0, tk.END):
            self.archive_dirs_listbox.insert(tk.END, directory)

    def _remove_archive_dir(self):
        selected_indices = self.archive_dirs_listbox.curselection()
        for i in reversed(selected_indices):
            self.archive_dirs_listbox.delete(i)

    def _add_exclude_name(self):
        name = simpledialog.askstring("Exclude Folder", "Enter folder name to ignore (e.g., 'thumbs'):")
        if name and name.strip():
            if name.strip() not in self.exclude_listbox.get(0, tk.END):
                self.exclude_listbox.insert(tk.END, name.strip())

    def _remove_exclude_name(self):
        selected_indices = self.exclude_listbox.curselection()
        for i in reversed(selected_indices):
            self.exclude_listbox.delete(i)

    def _select_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, directory)
            global OUTPUT_DIR
            OUTPUT_DIR = directory
            self.HITS_LOG = os.path.join(OUTPUT_DIR, "hits_log.csv")
            print(f"Output directory set to: {OUTPUT_DIR}. Hits log will be at: {self.HITS_LOG}")

    def _add_reference_person(self):
        person_name = simpledialog.askstring("Add Person", "Enter the name of the person:")
        if not person_name or person_name.strip() == "":
            return
        ref_paths = filedialog.askopenfilenames(title=f"Select Reference Images for {person_name}", filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")])
        if ref_paths:
            REFERENCE_IMAGES_CONFIG[person_name.strip()] = list(ref_paths)
            self._update_ref_images_listbox()
            self._show_selected_ref_images()

    def _update_ref_images_listbox(self):
        self.ref_images_listbox.delete(0, tk.END)
        for person, refs in REFERENCE_IMAGES_CONFIG.items():
            self.ref_images_listbox.insert(tk.END, f"{person}: {len(refs)} images")

    def _remove_reference_person(self):
        selected_indices = self.ref_images_listbox.curselection()
        if not selected_indices: return
        index = selected_indices[0]
        person_name = self.ref_images_listbox.get(index).split(":")[0].strip()
        if messagebox.askyesno("Remove Person", f"Are you sure you want to remove '{person_name}'?"):
            if person_name in REFERENCE_IMAGES_CONFIG:
                del REFERENCE_IMAGES_CONFIG[person_name]
                self._update_ref_images_listbox()
                self._clear_image_previews()

    def _show_selected_ref_images(self, event=None):
        self._clear_image_previews()
        selected_indices = self.ref_images_listbox.curselection()
        if not selected_indices: return
        index = selected_indices[0]
        person_name = self.ref_images_listbox.get(index).split(":")[0].strip()
        if person_name in REFERENCE_IMAGES_CONFIG:
            for i, ref_path in enumerate(REFERENCE_IMAGES_CONFIG[person_name]):
                try:
                    img = Image.open(ref_path)
                    img = ImageOps.exif_transpose(img)
                    img.thumbnail((100, 100))
                    img_tk = ImageTk.PhotoImage(img)
                    label = tk.Label(self.image_preview_frame, image=img_tk)
                    label.image = img_tk
                    label.pack(side=tk.LEFT, padx=5)
                except: pass

    def _clear_image_previews(self):
        for widget in self.image_preview_frame.winfo_children(): widget.destroy()

    def _get_current_config(self):
        global ARCHIVE_DIRS, OUTPUT_DIR, MODEL, DETECTOR, DIST_METRIC, MAX_DIST, EXCLUDED_FOLDER_NAMES, ENABLED_EXTENSIONS
        ARCHIVE_DIRS = list(self.archive_dirs_listbox.get(0, tk.END))
        EXCLUDED_FOLDER_NAMES = list(self.exclude_listbox.get(0, tk.END))
        OUTPUT_DIR = self.output_dir_entry.get().strip()
        self.HITS_LOG = os.path.join(OUTPUT_DIR, "hits_log.csv")
        MODEL = self.model_var.get()
        DETECTOR = self.detector_var.get()
        DIST_METRIC = self.metric_var.get()
        
        # Update enabled extensions
        for ext, var in self.ext_vars.items():
            ENABLED_EXTENSIONS[ext] = var.get()

        try: MAX_DIST = float(self.max_dist_entry.get().strip())
        except ValueError: return False
        
        if not ARCHIVE_DIRS or not OUTPUT_DIR:
            messagebox.showerror("Configuration Error", "Please check your Archive and Output directories.")
            return False
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return True

    def _save_config(self):
        if not self._get_current_config(): return
        config_data = {
            "archive_dirs": ARCHIVE_DIRS, "output_dir": OUTPUT_DIR,
            "reference_images_config": REFERENCE_IMAGES_CONFIG, "model": MODEL,
            "detector": DETECTOR, "distance_metric": DIST_METRIC, "max_dist": MAX_DIST,
            "excluded_folder_names": EXCLUDED_FOLDER_NAMES,
            "enabled_extensions": ENABLED_EXTENSIONS
        }
        try:
            with open("deepface_gui_config.pkl", "wb") as f: pickle.dump(config_data, f)
            print("Configuration saved successfully.")
        except Exception as e: print(f"⚠️ Error saving configuration: {e}")

    def _load_initial_config(self):
        global ARCHIVE_DIRS, OUTPUT_DIR, REFERENCE_IMAGES_CONFIG, MODEL, DETECTOR, DIST_METRIC, MAX_DIST, EXCLUDED_FOLDER_NAMES, ENABLED_EXTENSIONS
        try:
            with open("deepface_gui_config.pkl", "rb") as f:
                config_data = pickle.load(f)
            ARCHIVE_DIRS = config_data.get("archive_dirs", [])
            OUTPUT_DIR = config_data.get("output_dir", os.path.join(os.path.expanduser("~"), "DeepFaceHits"))
            REFERENCE_IMAGES_CONFIG = config_data.get("reference_images_config", {})
            MODEL = config_data.get("model", "ArcFace")
            DETECTOR = config_data.get("detector", "retinaface")
            DIST_METRIC = config_data.get("distance_metric", "cosine")
            MAX_DIST = config_data.get("max_dist", 0.28)
            EXCLUDED_FOLDER_NAMES = config_data.get("excluded_folder_names", ["$RECYCLE.BIN", "System Volume Information", ".git", "__pycache__"])
            loaded_exts = config_data.get("enabled_extensions", {})
            
            # Merge loaded extensions with defaults
            for ext, val in loaded_exts.items():
                ENABLED_EXTENSIONS[ext] = val

            # UI Populate
            for d in ARCHIVE_DIRS: self.archive_dirs_listbox.insert(tk.END, d)
            for d in EXCLUDED_FOLDER_NAMES: self.exclude_listbox.insert(tk.END, d)
            
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, OUTPUT_DIR)
            self._update_ref_images_listbox()
            self.model_var.set(MODEL)
            self.detector_var.set(DETECTOR)
            self.metric_var.set(DIST_METRIC)
            self.max_dist_entry.delete(0, tk.END)
            self.max_dist_entry.insert(0, str(MAX_DIST))
            
            # Set checkboxes
            for ext, var in self.ext_vars.items():
                if ext in ENABLED_EXTENSIONS:
                    var.set(ENABLED_EXTENSIONS[ext])

            self.HITS_LOG = os.path.join(OUTPUT_DIR, "hits_log.csv")
        except: print("No saved configuration found. Starting with default values.")

    # --- Analysis Methods ---

    def _start_analysis(self):
        if self.running_thread and self.running_thread.is_alive():
            messagebox.showinfo("Info", "Analysis is already running.")
            return
        if not self._get_current_config(): return
        if not REFERENCE_IMAGES_CONFIG:
            messagebox.showerror("Config Error", "You need at least one Reference Person to run an analysis.")
            return

        print("\n--- Starting Analysis ---")
        self.stop_event.clear()
        self.running_thread = threading.Thread(target=self._run_analysis_in_thread, args=(False,))
        self.running_thread.daemon = True
        self.running_thread.start()

    def _start_indexing_only(self):
        if self.running_thread and self.running_thread.is_alive():
            messagebox.showinfo("Info", "A process is already running.")
            return
        if not self._get_current_config(): return
        
        print("\n--- Starting Indexing Only ---")
        self.stop_event.clear()
        self.running_thread = threading.Thread(target=self._run_analysis_in_thread, args=(True,))
        self.running_thread.daemon = True
        self.running_thread.start()

    def _stop_analysis(self):
        if self.running_thread and self.running_thread.is_alive():
            self.stop_event.set()
            self._update_status("Stopping...")
            print("\n--- Stopping Analysis ---")
        else: messagebox.showinfo("Info", "No analysis is currently running.")

    def _precompute_reference_embeddings(self):
        """Calculates and returns a list of dictionaries containing reference embeddings."""
        live_refs = []
        print("🔄 Pre-calculating reference embeddings for live matching...")
        self._update_status("Pre-calculating reference embeddings...")
        
        for person, refs in REFERENCE_IMAGES_CONFIG.items():
            for ref_path in refs:
                if not os.path.exists(ref_path): continue
                try:
                    img_input = self._load_image_fixed(ref_path)
                    ref_reps = DeepFace.represent(img_path=img_input, model_name=MODEL, detector_backend=DETECTOR, enforce_detection=False)
                    if ref_reps:
                        emb = np.array(ref_reps[0]["embedding"])
                        if DIST_METRIC == "cosine":
                            # Pre-normalize to speed up math inside loop
                            emb = emb / np.linalg.norm(emb)
                        live_refs.append({
                            "person": person,
                            "ref_path": ref_path,
                            "embedding": emb
                        })
                except Exception as e:
                    print(f"⚠️ Error loading reference {os.path.basename(ref_path)}: {e}")
        
        print(f"✅ Loaded {len(live_refs)} reference embeddings.")
        return live_refs

    def _handle_live_match(self, archive_path, archive_emb, live_refs, archive_dir):
        """Compare a single new archive embedding against all references and copy matches."""
        
        # Pre-normalize archive embedding if cosine (references are already normalized)
        if DIST_METRIC == "cosine":
            norm = np.linalg.norm(archive_emb)
            if norm == 0: return
            archive_emb_norm = archive_emb / norm
        
        for ref in live_refs:
            ref_emb = ref["embedding"]
            
            if DIST_METRIC == "cosine":
                distance = 1 - np.dot(archive_emb_norm, ref_emb)
            else:
                distance = np.linalg.norm(archive_emb - ref_emb)
            
            # --- Check Logic ---
            if distance <= MAX_DIST:
                # Filter Exact Duplicates (Self-Matches)
                if distance < 1e-6: continue
                
                person = ref["person"]
                print(f"🔥 LIVE MATCH FOUND! {person} -> {os.path.basename(archive_path)} (dist: {distance:.4f})")
                
                # Copy File Logic
                person_dir = os.path.join(OUTPUT_DIR, person)
                os.makedirs(person_dir, exist_ok=True)
                dest_path = os.path.join(person_dir, os.path.basename(archive_path))
                
                # Avoid expensive Copy if file exists
                if not os.path.exists(dest_path):
                    try:
                        shutil.copy2(archive_path, dest_path)
                        # Log to CSV (Memory Only for now, or append mode)
                        new_row = {
                            "person": person, "archive_dir": archive_dir, 
                            "identity": archive_path, "distance": distance, "copied_path": dest_path
                        }
                        # Append to in-memory log
                        self.hits_log_df = pd.concat([self.hits_log_df, pd.DataFrame([new_row])], ignore_index=True)
                        # Safe atomic save of CSV
                        self.hits_log_df.to_csv(self.HITS_LOG, index=False)
                    except Exception as e: print(f"⚠️ Live copy failed: {e}")

    def _run_analysis_in_thread(self, index_only=False):
        try:
            self.hits_log_df = self._load_hits_log()
            
            # 0. Prepare Live References (if we are not just indexing blind)
            live_references = []
            if not index_only and not self.skip_indexing_var.get():
                live_references = self._precompute_reference_embeddings()

            # 1. Load/Build Embeddings + Live Match
            all_archive_embeddings = []
            for archive_dir in ARCHIVE_DIRS:
                if self.stop_event.is_set(): break
                
                should_index = True if index_only else (not self.skip_indexing_var.get())
                
                if should_index:
                    print(f"\n🧠 Checking/Updating index for: {archive_dir}")
                    # Pass live_references here for real-time matching
                    self._incremental_index(MODEL, archive_dir, live_references=live_references)
                    self._update_timer("") 
                else:
                    print(f"\n🧠 Using existing index (Read-Only) for: {archive_dir}")

                if index_only: continue

                # Load index for final sweep (or if skipping index)
                index_path = os.path.join(archive_dir, f"representations_{MODEL.lower()}.pkl")
                if os.path.exists(index_path):
                    try:
                        self._update_status(f"Loading index: {os.path.basename(archive_dir)}")
                        with open(index_path, 'rb') as f:
                            archive_df = pickle.load(f)
                            if "status" not in archive_df.columns: archive_df["status"] = "ok"
                            filtered = archive_df[archive_df["status"] == "ok"][["identity", "embedding"]]
                            filtered["archive_dir"] = archive_dir 
                            if not filtered.empty:
                                all_archive_embeddings.append(filtered)
                    except Exception as e: print(f"⚠️ Failed to load index: {e}")

            if index_only:
                print(f"\n{'=' * 60}\n🎉 INDEXING COMPLETE\n{'=' * 60}")
                self._update_status("Indexing Complete.")
                self._update_timer("")
                return

            if not all_archive_embeddings:
                print("❌ No valid embeddings found.")
                self._update_status("Error: No embeddings found")
                return
            
            # 2. Final Sweep (Catches matches if 'Skip Indexing' was used, or confirms live matches)
            # Only runs if we actually loaded data
            print(f"\n{'=' * 60}\n🔍 Performing Final Verification Sweep\n{'=' * 60}")
            
            unified_df = pd.concat(all_archive_embeddings, ignore_index=True)
            archive_embeddings_np = np.array(unified_df['embedding'].tolist())
            
            total_copied = 0
            for person, refs in REFERENCE_IMAGES_CONFIG.items():
                if self.stop_event.is_set(): break
                
                potential_matches_df = pd.DataFrame()
                for ref in refs:
                    if self.stop_event.is_set(): break
                    if not os.path.exists(ref): continue

                    try:
                        img_input = self._load_image_fixed(ref)
                        ref_reps = DeepFace.represent(img_path=img_input, model_name=MODEL, detector_backend=DETECTOR, enforce_detection=False)
                        if not ref_reps: continue
                        
                        ref_embedding = np.array(ref_reps[0]["embedding"])

                        if DIST_METRIC == "cosine":
                            ref_embedding_norm = ref_embedding / np.linalg.norm(ref_embedding)
                            archive_embeddings_norm = archive_embeddings_np / np.linalg.norm(archive_embeddings_np, axis=1, keepdims=True)
                            distances = 1 - np.dot(archive_embeddings_norm, ref_embedding_norm)
                        else:
                            distances = np.linalg.norm(archive_embeddings_np - ref_embedding, axis=1)

                        matches_indices = np.where(distances <= MAX_DIST)[0]
                        
                        if len(matches_indices) > 0:
                            matches = unified_df.iloc[matches_indices].copy()
                            matches["distance"] = distances[matches_indices]
                            potential_matches_df = pd.concat([potential_matches_df, matches], ignore_index=True)
                    except: pass

                if not potential_matches_df.empty:
                    self.hits_log_df, copied = self._copy_hits_for_archive(potential_matches_df, person, "multi-source", self.hits_log_df)
                    total_copied += copied

            if not self.stop_event.is_set():
                print(f"\n{'=' * 60}\n🎉 COMPLETE\n{'=' * 60}")
                self._update_status("Analysis Complete.")

        except Exception as e:
            print(f"\n--- FATAL ERROR IN ANALYSIS ---: {e}")
            self._update_status("Fatal Error (see console)")
            import traceback
            print(traceback.format_exc())
        finally:
            self.running_thread = None
            self._update_timer("")

    def _calculate_min_distances_optimized(self):
        if not self._get_current_config(): return
        if not REFERENCE_IMAGES_CONFIG:
            messagebox.showerror("Config Error", "No Reference images.")
            return

        self.min_dist_display.delete(1.0, tk.END)
        print("\n--- Calculating Minimum Distances ---")
        
        all_archive_embeddings = [] 
        for archive_dir in ARCHIVE_DIRS:
            index_path = os.path.join(archive_dir, f"representations_{MODEL.lower()}.pkl")
            if os.path.exists(index_path):
                try:
                    with open(index_path, 'rb') as f:
                        archive_df = pickle.load(f)
                        if "status" not in archive_df.columns: archive_df["status"] = "ok" 
                        filtered = archive_df[archive_df["status"] == "ok"][["identity", "embedding"]]
                        if not filtered.empty: all_archive_embeddings.append(filtered)
                except: pass
        
        if not all_archive_embeddings:
            messagebox.showerror("Error", "No embeddings indexes found.")
            return

        unified_embeddings_df = pd.concat(all_archive_embeddings, ignore_index=True)
        archive_embeddings_np = np.array(unified_embeddings_df['embedding'].tolist())

        self.min_dist_display.tag_config("hyperlink", foreground="blue", underline=1, font=("Segoe UI", 9, "bold"))
        self.min_dist_display.tag_bind("hyperlink", "<Enter>", lambda e: self.min_dist_display.config(cursor="hand2"))
        self.min_dist_display.tag_bind("hyperlink", "<Leave>", lambda e: self.min_dist_display.config(cursor="arrow"))

        for person, refs in REFERENCE_IMAGES_CONFIG.items():
            for ref_path in refs:
                if not os.path.exists(ref_path): continue
                try:
                    self._update_status(f"Calculating min dist for: {os.path.basename(ref_path)}")
                    img_input = self._load_image_fixed(ref_path)
                    ref_reps = DeepFace.represent(img_path=img_input, model_name=MODEL, detector_backend=DETECTOR, enforce_detection=False)
                    if not ref_reps: continue
                    ref_embedding = np.array(ref_reps[0]["embedding"])

                    if DIST_METRIC == "cosine":
                        ref_embedding_norm = ref_embedding / np.linalg.norm(ref_embedding)
                        archive_embeddings_norm = archive_embeddings_np / np.linalg.norm(archive_embeddings_np, axis=1, keepdims=True)
                        distances = 1 - np.dot(archive_embeddings_norm, ref_embedding_norm)
                    else:
                        distances = np.linalg.norm(archive_embeddings_np - ref_embedding, axis=1)
                    
                    distances[distances < 1e-6] = np.inf
                    min_index = np.argmin(distances)
                    lowest_dist_for_ref = distances[min_index]
                    
                    if lowest_dist_for_ref == np.inf:
                        self.min_dist_display.insert(tk.END, f"Ref: {os.path.basename(ref_path)} ({person})\nResult: Exact matches only.\n\n")
                        continue

                    closest_match_path = unified_embeddings_df.iloc[min_index]["identity"]
                    closest_match_path = os.path.abspath(closest_match_path)

                    self.min_dist_display.insert(tk.END, f"Ref: {os.path.basename(ref_path)} ({person})\n")
                    self.min_dist_display.insert(tk.END, f"  Dist: {lowest_dist_for_ref:.4f}\n")
                    self.min_dist_display.insert(tk.END, f"  Path: {closest_match_path}\n")
                    unique_tag = f"link_{int(time.time()*1000)}_{min_index}"
                    self.min_dist_display.insert(tk.END, "  👉 [ OPEN IMAGE ]\n\n", (unique_tag, "hyperlink"))
                    self.min_dist_display.tag_bind(unique_tag, "<Button-1>", lambda e, p=closest_match_path: self._open_file_from_link(p))

                except Exception as e: print(f"Error: {e}")
        print("Minimum distances calculation complete.")
        self.min_dist_display.see(tk.END)
        self._update_status("Min Distance Calc Complete")

    def _load_hits_log(self):
        if os.path.exists(self.HITS_LOG):
            try: return pd.read_csv(self.HITS_LOG)
            except: pass
        return pd.DataFrame(columns=["person", "archive_dir", "identity", "distance", "copied_path"])

    def _atomic_pickle_save(self, df, path):
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, 'wb') as f: pickle.dump(df, f)
            os.replace(tmp_path, path)
        except: 
            if os.path.exists(tmp_path): os.remove(tmp_path)

    def _incremental_index(self, model_name, db_path, live_references=None):
        """Scan folder, calc embeddings. If live_references is provided, check matches immediately."""
        index_path = os.path.join(db_path, f"representations_{model_name.lower()}.pkl")
        bak_path = index_path + ".bak"

        if os.path.exists(index_path):
            try: shutil.copy2(index_path, bak_path)
            except: pass

        existing = pd.DataFrame()
        processed_paths = set()
        
        files_to_try = [index_path, index_path + ".tmp", bak_path]
        for fp in files_to_try:
            if os.path.exists(fp):
                try:
                    with open(fp, 'rb') as f: existing = pickle.load(f)
                    if not existing.empty:
                        if 'identity' in existing.columns: processed_paths = set(existing['identity'].tolist())
                        break 
                except: pass
        
        if not existing.empty:
            if "status" not in existing.columns: existing["status"] = "ok"
            if "error" not in existing.columns: existing["error"] = None

        df = existing.copy()
        initial_df_len = len(df)
        last_save_time = time.time()

        # Build dynamic list of allowed extensions
        allowed_exts = tuple(ext for ext, enabled in ENABLED_EXTENSIONS.items() if enabled)

        def maybe_checkpoint(force=False):
            nonlocal last_save_time, df
            now = time.time()
            remaining = CHECKPOINT_INTERVAL_SECONDS - (now - last_save_time)
            if remaining < 0: remaining = 0
            mins, secs = divmod(int(remaining), 60)
            self._update_timer(f"Next Save: {mins:02d}:{secs:02d}")
            
            if force or (now - last_save_time) >= CHECKPOINT_INTERVAL_SECONDS:
                self._update_timer("Saving Checkpoint...")
                self._atomic_pickle_save(df, index_path)
                last_save_time = now
                print(f"💾 checkpoint saved ({len(df)} entries)")

        for root, dirs, files in os.walk(db_path):
            # --- EXCLUSION LOGIC ---
            # Modify dirs in-place to prevent walking into excluded folders
            dirs[:] = [d for d in dirs if d not in EXCLUDED_FOLDER_NAMES]
            
            files.sort() 
            for file in files:
                if self.stop_event.is_set(): 
                    print("🛑 Stop requested. Saving current progress...")
                    maybe_checkpoint(force=True) 
                    return df

                # --- EXTENSION FILTER LOGIC ---
                if not file.lower().endswith(allowed_exts): continue
                
                img_path = os.path.join(root, file)
                
                self._update_status(f"Indexing: {img_path}")
                maybe_checkpoint(force=False)
                
                if img_path in processed_paths: continue

                try:
                    reps = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=False, detector_backend=DETECTOR)
                    if isinstance(reps, list) and len(reps) > 0:
                        emb = reps[0]["embedding"]
                        new_row = {"identity": img_path, "embedding": emb, "status": "ok", "error": None}
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                        processed_paths.add(img_path)
                        
                        # --- LIVE MATCHING HOOK ---
                        if live_references:
                            self._handle_live_match(img_path, np.array(emb), live_references, db_path)
                        # --------------------------

                except Exception as e:
                    msg = str(e)
                    print(f"⚠️ Skipping {img_path}: {msg}")
                    new_row = {"identity": img_path, "embedding": None, "status": "failed", "error": msg}
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    processed_paths.add(img_path)

        if len(df) > initial_df_len or not existing.equals(df): maybe_checkpoint(force=True)
        print(f"✅ index complete: {len(df)} entries saved")
        return df

    def _copy_hits_for_archive(self, df, person, archive_dir, hits_log_df):
        if df is None or df.empty: return hits_log_df, 0
        df = df.sort_values(by="distance").drop_duplicates(subset=["identity"])
        hits = df[df["distance"] <= MAX_DIST].copy()
        if hits.empty: return hits_log_df, 0

        person_dir = os.path.join(OUTPUT_DIR, person)
        os.makedirs(person_dir, exist_ok=True)
        already_copied = set(hits_log_df["identity"]) if not hits_log_df.empty else set()
        copied = 0
        new_rows = []

        for _, row in hits.iterrows():
            if self.stop_event.is_set(): break 
            p = row["identity"]
            dist = row["distance"]
            src_arch = row.get("archive_dir", archive_dir)

            if p in already_copied: continue 
            
            if dist < 1e-6: continue # Skip exact

            try:
                if os.path.exists(p):
                    dest_path = os.path.join(person_dir, os.path.basename(p))
                    # Avoid overwrite if live matching already grabbed it
                    if not os.path.exists(dest_path):
                        shutil.copy2(p, dest_path)
                        copied += 1
                        print(f"✅ Copied: {os.path.basename(p)} (dist: {dist:.4f})")
                    
                    new_rows.append({"person": person, "archive_dir": src_arch, "identity": p, "distance": dist, "copied_path": dest_path})
                else: print(f"⚠️ Missing file (skipped copy): {p}")
            except Exception as e: print(f"⚠️ Copy failed for {p}: {e}")

        if new_rows:
            hits_log_df = pd.concat([hits_log_df, pd.DataFrame(new_rows)], ignore_index=True)
            hits_log_df.to_csv(self.HITS_LOG, index=False)
        return hits_log_df, copied

if __name__ == "__main__":
    app = FaceFinderGUI()
    app.mainloop()
