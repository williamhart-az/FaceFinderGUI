# /// script
# dependencies = [
#     "deepface",
#     "pandas",
#     "numpy",
#     "pillow",
#     "tf-keras",
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

# --- Configuration ---
ARCHIVE_DIRS = []
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "DeepFaceHits") 
REFERENCE_IMAGES_CONFIG = {} 

MODEL = "ArcFace"
DETECTOR = "retinaface"
DIST_METRIC = "cosine"
MAX_DIST = 0.28

CHECKPOINT_INTERVAL_SECONDS = 5 * 60  # 5 minutes

# ========== TOOLTIP CLASS ==========
class ToolTip(object):
    """
    Creates a tooltip for a given widget as the mouse hovers above it.
    """
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
        # 500ms delay before showing tooltip so it doesn't flash annoying users
        self.id = self.widget.after(500, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + cy + self.widget.winfo_rooty() + 25
        
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1) # Remove window decorations
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
        self.title("DeepFace GUI Face Finder")
        self.geometry("1200x850")

        self.hits_log_df = pd.DataFrame()
        self.running_thread = None
        self.stop_event = threading.Event() 
        
        # Toggle for skipping index generation
        self.skip_indexing_var = tk.BooleanVar(value=True) 

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.HITS_LOG = os.path.join(OUTPUT_DIR, "hits_log.csv")

        self._create_menu() # Create the Help Menu
        self._create_widgets()
        self._load_initial_config() 

    def _create_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Manual / Model Guide", command=self._show_manual)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "FaceFinder GUI v2.0\nPowered by DeepFace"))

    def _add_tooltip(self, widget, text):
        """Helper to attach a tooltip to a widget"""
        ToolTip(widget, text)

    def _create_widgets(self):
        # --- Frame for Configuration ---
        config_frame = tk.LabelFrame(self, text="Configuration", padx=10, pady=10)
        config_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Archive Directories
        tk.Label(config_frame, text="Archive Directories:").grid(row=0, column=0, sticky="w", pady=2)
        self.archive_dirs_listbox = tk.Listbox(config_frame, height=3, width=70)
        self.archive_dirs_listbox.grid(row=0, column=1, columnspan=2, pady=2)
        self._add_tooltip(self.archive_dirs_listbox, "The folders containing the massive photo collections you want to search through.")
        
        btn_add_arc = tk.Button(config_frame, text="Add", command=self._add_archive_dir)
        btn_add_arc.grid(row=0, column=3, padx=2)
        self._add_tooltip(btn_add_arc, "Add a new folder to search.")
        
        tk.Button(config_frame, text="Remove", command=self._remove_archive_dir).grid(row=0, column=4, padx=2)

        # Output Directory
        tk.Label(config_frame, text="Output Directory:").grid(row=1, column=0, sticky="w", pady=2)
        self.output_dir_entry = tk.Entry(config_frame, width=70)
        self.output_dir_entry.insert(0, OUTPUT_DIR) 
        self.output_dir_entry.grid(row=1, column=1, columnspan=2, pady=2)
        self._add_tooltip(self.output_dir_entry, "The folder where matched photos will be copied to.")
        
        tk.Button(config_frame, text="Browse", command=self._select_output_dir).grid(row=1, column=3, padx=2)

        # Reference Images
        tk.Label(config_frame, text="Reference Images (Person: Paths):").grid(row=2, column=0, sticky="w", pady=2)
        self.ref_images_listbox = tk.Listbox(config_frame, height=4, width=70)
        self.ref_images_listbox.grid(row=2, column=1, columnspan=2, pady=2)
        self._add_tooltip(self.ref_images_listbox, "List of people and their reference photos to find.")
        
        tk.Button(config_frame, text="Add Person/Refs", command=self._add_reference_person).grid(row=2, column=3, padx=2)
        tk.Button(config_frame, text="Remove Person", command=self._remove_reference_person).grid(row=2, column=4, padx=2)

        # DeepFace Model/Detector/Metric
        tk.Label(config_frame, text="Model:").grid(row=3, column=0, sticky="w", pady=2)
        self.model_var = tk.StringVar(self)
        self.model_var.set(MODEL)
        models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
        om_model = tk.OptionMenu(config_frame, self.model_var, *models)
        om_model.grid(row=3, column=1, sticky="ew", pady=2)
        self._add_tooltip(om_model, "The 'Brain'. Determines accuracy.\nRecommend: ArcFace (Best) or Facenet512.")

        tk.Label(config_frame, text="Detector:").grid(row=3, column=2, sticky="w", pady=2)
        self.detector_var = tk.StringVar(self)
        self.detector_var.set(DETECTOR)
        detectors = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe", "yolov8"]
        om_detect = tk.OptionMenu(config_frame, self.detector_var, *detectors)
        om_detect.grid(row=3, column=3, sticky="ew", pady=2)
        self._add_tooltip(om_detect, "The 'Eyes'. Finds faces in the image.\nRecommend: RetinaFace (Most accurate for side/small faces).")

        tk.Label(config_frame, text="Distance Metric:").grid(row=4, column=0, sticky="w", pady=2)
        self.metric_var = tk.StringVar(self)
        self.metric_var.set(DIST_METRIC)
        metrics = ["cosine", "euclidean", "euclidean_l2"]
        om_metric = tk.OptionMenu(config_frame, self.metric_var, *metrics)
        om_metric.grid(row=4, column=1, sticky="ew", pady=2)
        self._add_tooltip(om_metric, "The 'Ruler'. How math compares faces.\nRecommend: Cosine.")

        tk.Label(config_frame, text="Max Distance:").grid(row=4, column=2, sticky="w", pady=2)
        self.max_dist_entry = tk.Entry(config_frame, width=10)
        self.max_dist_entry.insert(0, str(MAX_DIST))
        self.max_dist_entry.grid(row=4, column=3, sticky="w", pady=2)
        self._add_tooltip(self.max_dist_entry, "Threshold for a match. Lower is stricter.\nArcFace/Cosine typical: 0.30 - 0.45.")

        # --- Frame for Actions and Results ---
        action_results_frame = tk.Frame(self, padx=10, pady=10)
        action_results_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Checkbox for Skipping Index
        cb_skip = tk.Checkbutton(
            action_results_frame, 
            text="Skip Indexing (Use existing PKL - Pure Math Mode)", 
            variable=self.skip_indexing_var
        )
        cb_skip.pack(anchor="w", pady=(0, 5))
        self._add_tooltip(cb_skip, "If checked: Uses previously saved analysis files.\nVery fast, but won't find newly added files.")

        # Buttons
        button_frame = tk.Frame(action_results_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        btn_run = tk.Button(button_frame, text="Run Analysis", command=self._start_analysis, bg="green", fg="white")
        btn_run.pack(side=tk.LEFT, padx=5)
        self._add_tooltip(btn_run, "Start the full search process.\nCopies matching files to Output Directory.")

        btn_stop = tk.Button(button_frame, text="Stop Analysis", command=self._stop_analysis, bg="red", fg="white")
        btn_stop.pack(side=tk.LEFT, padx=5)
        
        # Optimized Calculation Button
        btn_calc = tk.Button(button_frame, text="Calculate Min Distances", command=self._calculate_min_distances_optimized)
        btn_calc.pack(side=tk.LEFT, padx=5)
        self._add_tooltip(btn_calc, "Quickly checks your reference photos against the existing index.\nGood for testing which reference photo is best.")

        tk.Button(button_frame, text="Save Configuration", command=self._save_config).pack(side=tk.LEFT, padx=5)

        # Min Distance Display
        tk.Label(action_results_frame, text="Min Distances for Reference Images:").pack(anchor="w", pady=5)
        self.min_dist_display = scrolledtext.ScrolledText(action_results_frame, height=8, wrap=tk.WORD)
        self.min_dist_display.pack(fill=tk.X, pady=2)

        # Console Output
        tk.Label(action_results_frame, text="Console Output:").pack(anchor="w", pady=5)
        self.console_output = scrolledtext.ScrolledText(action_results_frame, height=10, wrap=tk.WORD)
        self.console_output.pack(fill=tk.BOTH, expand=True)

        # Reference Image Previews
        tk.Label(action_results_frame, text="Reference Image Previews:").pack(anchor="w", pady=5)
        self.image_preview_frame = tk.Frame(action_results_frame)
        self.image_preview_frame.pack(fill=tk.X, pady=5)

        self.ref_images_listbox.bind("<<ListboxSelect>>", self._show_selected_ref_images)

        # Redirect stdout
        self.console_output.tag_config('info', foreground='black')
        self.console_output.tag_config('warning', foreground='orange')
        self.console_output.tag_config('error', foreground='red')
        self.console_output.tag_config('success', foreground='green')

        import sys
        sys.stdout = self
        sys.stderr = self

    def _show_manual(self):
        """Opens a Toplevel window with the User Manual"""
        manual_win = tk.Toplevel(self)
        manual_win.title("User Manual & Guide")
        manual_win.geometry("800x600")
        
        txt = scrolledtext.ScrolledText(manual_win, wrap=tk.WORD, padx=10, pady=10)
        txt.pack(fill=tk.BOTH, expand=True)
        
        # Formatting tags
        txt.tag_config("h1", font=("Segoe UI", 14, "bold"), foreground="#2c3e50")
        txt.tag_config("h2", font=("Segoe UI", 12, "bold"), foreground="#2980b9")
        txt.tag_config("bold", font=("Segoe UI", 10, "bold"))
        
        # Content
        txt.insert(tk.END, "FaceFinder GUI Manual\n\n", "h1")
        
        txt.insert(tk.END, "1. Models (The 'Brain')\n", "h2")
        txt.insert(tk.END, "The Model determines how accurate the recognition is.\n\n")
        txt.insert(tk.END, "ArcFace (Recommended): ", "bold")
        txt.insert(tk.END, "State-of-the-art accuracy. Best for separating different people.\n")
        txt.insert(tk.END, "Facenet512: ", "bold")
        txt.insert(tk.END, "Google's model. Very reliable, slightly faster than ArcFace.\n")
        txt.insert(tk.END, "VGG-Face: ", "bold")
        txt.insert(tk.END, "Older, reliable, but computationally heavy.\n")
        txt.insert(tk.END, "SFace: ", "bold")
        txt.insert(tk.END, "Fastest, designed for mobile. Lower accuracy.\n\n")
        
        txt.insert(tk.END, "2. Detectors (The 'Eyes')\n", "h2")
        txt.insert(tk.END, "The Detector finds the face in the photo before the Model analyzes it.\n\n")
        txt.insert(tk.END, "RetinaFace (Recommended): ", "bold")
        txt.insert(tk.END, "Amazing at finding small, blurry, or sideways faces. Slower, but best for archives.\n")
        txt.insert(tk.END, "MediaPipe: ", "bold")
        txt.insert(tk.END, "Google's solution. Very fast. Good balance of speed/accuracy.\n")
        txt.insert(tk.END, "OpenCV/SSD: ", "bold")
        txt.insert(tk.END, "Very fast but basic. Only finds faces looking directly at the camera.\n\n")
        
        txt.insert(tk.END, "3. Distance Metrics\n", "h2")
        txt.insert(tk.END, "Cosine (Recommended): ", "bold")
        txt.insert(tk.END, "Measures the angle between faces. Best for lighting variations.\n")
        txt.insert(tk.END, "Euclidean L2: ", "bold")
        txt.insert(tk.END, "Measures straight-line distance.\n\n")
        
        txt.insert(tk.END, "4. Workflow Tips\n", "h2")
        txt.insert(tk.END, "• Use 'Calculate Min Distances' to check if your reference photo is good.\n")
        txt.insert(tk.END, "• If you see a distance < 0.40, your reference is working well.\n")
        txt.insert(tk.END, "• Check 'Skip Indexing' once you have run the analysis once to speed up future searches.\n")
        
        txt.configure(state=tk.DISABLED) # Make read-only

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

    def flush(self):
        pass

    # --- Helper Methods ---
    
    def _load_image_fixed(self, path):
        """Loads an image, fixes EXIF rotation, ensures RGB, and returns a BGR numpy array for DeepFace."""
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            
            # CRITICAL FIX: Convert to RGB to remove Alpha channel (Transparency)
            img = img.convert("RGB") 
            
            img_np = np.array(img)
            # Convert RGB to BGR for DeepFace
            img_np = img_np[:, :, ::-1] 
            return img_np
        except Exception as e:
            print(f"⚠️ Error processing image {os.path.basename(path)}: {e}")
            return path 

    def _open_file_from_link(self, filepath):
        """Opens a file using the OS default viewer."""
        print(f"🖱️ Link clicked! Attempting to open: {filepath}")
        if not os.path.exists(filepath):
            messagebox.showerror("File Not Found", f"Could not find file at:\n{filepath}")
            return
        try:
            if platform.system() == 'Darwin':       # macOS
                subprocess.call(('open', filepath))
            elif platform.system() == 'Windows':    # Windows
                os.startfile(filepath)
            else:                                   # Linux
                subprocess.call(('xdg-open', filepath))
        except Exception as e:
            print(f"⚠️ Standard open failed: {e}")
            try:
                folder = os.path.dirname(filepath)
                if platform.system() == 'Windows':
                    os.startfile(folder)
                else:
                    print("Try manually navigating to the folder.")
            except:
                pass

    def _add_archive_dir(self):
        directory = filedialog.askdirectory()
        if directory and directory not in self.archive_dirs_listbox.get(0, tk.END):
            self.archive_dirs_listbox.insert(tk.END, directory)

    def _remove_archive_dir(self):
        selected_indices = self.archive_dirs_listbox.curselection()
        for i in reversed(selected_indices):
            self.archive_dirs_listbox.delete(i)

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
        ref_paths = filedialog.askopenfilenames(
            title=f"Select Reference Images for {person_name}",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")]
        )
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
        if not selected_indices:
            messagebox.showinfo("Selection Error", "Please select a person to remove.")
            return
        index = selected_indices[0]
        item_text = self.ref_images_listbox.get(index)
        person_name = item_text.split(":")[0].strip()
        if messagebox.askyesno("Remove Person", f"Are you sure you want to remove '{person_name}' and their reference images?"):
            if person_name in REFERENCE_IMAGES_CONFIG:
                del REFERENCE_IMAGES_CONFIG[person_name]
                self._update_ref_images_listbox()
                self._clear_image_previews()

    def _show_selected_ref_images(self, event=None):
        self._clear_image_previews()
        selected_indices = self.ref_images_listbox.curselection()
        if not selected_indices:
            return
        index = selected_indices[0]
        item_text = self.ref_images_listbox.get(index)
        person_name = item_text.split(":")[0].strip()
        if person_name in REFERENCE_IMAGES_CONFIG:
            refs = REFERENCE_IMAGES_CONFIG[person_name]
            for i, ref_path in enumerate(refs):
                try:
                    img = Image.open(ref_path)
                    img = ImageOps.exif_transpose(img) # Fix rotation for preview
                    img.thumbnail((100, 100))
                    img_tk = ImageTk.PhotoImage(img)
                    label = tk.Label(self.image_preview_frame, image=img_tk)
                    label.image = img_tk
                    label.pack(side=tk.LEFT, padx=5)
                except Exception as e:
                    print(f"⚠️ Could not load image {ref_path}: {e}")
                    tk.Label(self.image_preview_frame, text=f"Error loading {os.path.basename(ref_path)}").pack(side=tk.LEFT, padx=5)

    def _clear_image_previews(self):
        for widget in self.image_preview_frame.winfo_children():
            widget.destroy()

    def _get_current_config(self):
        global ARCHIVE_DIRS, OUTPUT_DIR, MODEL, DETECTOR, DIST_METRIC, MAX_DIST
        ARCHIVE_DIRS = list(self.archive_dirs_listbox.get(0, tk.END))
        OUTPUT_DIR = self.output_dir_entry.get().strip()
        self.HITS_LOG = os.path.join(OUTPUT_DIR, "hits_log.csv")
        MODEL = self.model_var.get()
        DETECTOR = self.detector_var.get()
        DIST_METRIC = self.metric_var.get()
        try:
            MAX_DIST = float(self.max_dist_entry.get().strip())
        except ValueError:
            messagebox.showerror("Input Error", "Max Distance must be a valid number.")
            return False
        if not ARCHIVE_DIRS:
            messagebox.showerror("Configuration Error", "Please add at least one Archive Directory.")
            return False
        if not OUTPUT_DIR:
            messagebox.showerror("Configuration Error", "Please specify an Output Directory.")
            return False
        if not REFERENCE_IMAGES_CONFIG:
            messagebox.showerror("Configuration Error", "Please add at least one Reference Person with images.")
            return False
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return True

    def _save_config(self):
        if not self._get_current_config():
            return
        config_data = {
            "archive_dirs": ARCHIVE_DIRS,
            "output_dir": OUTPUT_DIR,
            "reference_images_config": REFERENCE_IMAGES_CONFIG,
            "model": MODEL,
            "detector": DETECTOR,
            "distance_metric": DIST_METRIC,
            "max_dist": MAX_DIST
        }
        try:
            with open("deepface_gui_config.pkl", "wb") as f:
                pickle.dump(config_data, f)
            print("Configuration saved successfully.")
        except Exception as e:
            print(f"⚠️ Error saving configuration: {e}")

    def _load_initial_config(self):
        global ARCHIVE_DIRS, OUTPUT_DIR, REFERENCE_IMAGES_CONFIG, MODEL, DETECTOR, DIST_METRIC, MAX_DIST
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
            for d in ARCHIVE_DIRS:
                self.archive_dirs_listbox.insert(tk.END, d)
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, OUTPUT_DIR)
            self._update_ref_images_listbox()
            self.model_var.set(MODEL)
            self.detector_var.set(DETECTOR)
            self.metric_var.set(DIST_METRIC)
            self.max_dist_entry.delete(0, tk.END)
            self.max_dist_entry.insert(0, str(MAX_DIST))
            self.HITS_LOG = os.path.join(OUTPUT_DIR, "hits_log.csv")
            if self.ref_images_listbox.size() > 0:
                self.ref_images_listbox.selection_set(0)
                self.ref_images_listbox.event_generate("<<ListboxSelect>>")
            print("Configuration loaded from deepface_gui_config.pkl")
        except FileNotFoundError:
            print("No saved configuration found. Starting with default values.")
        except Exception as e:
            print(f"⚠️ Error loading configuration: {e}")

    # --- Analysis Methods ---

    def _start_analysis(self):
        if self.running_thread and self.running_thread.is_alive():
            messagebox.showinfo("Info", "Analysis is already running.")
            return
        if not self._get_current_config():
            return
        print("\n--- Starting Analysis ---")
        self.stop_event.clear()
        self.running_thread = threading.Thread(target=self._run_analysis_in_thread)
        self.running_thread.daemon = True
        self.running_thread.start()

    def _stop_analysis(self):
        if self.running_thread and self.running_thread.is_alive():
            self.stop_event.set()
            print("\n--- Stopping Analysis (may take a moment to finish current operation) ---")
        else:
            messagebox.showinfo("Info", "No analysis is currently running.")

    def _run_analysis_in_thread(self):
        try:
            self.hits_log_df = self._load_hits_log()
            total_copied = 0
            
            # 1. Load ALL Embeddings first (One time)
            all_archive_embeddings = []
            for archive_dir in ARCHIVE_DIRS:
                if self.stop_event.is_set(): break
                
                # Should we update the index first?
                if not self.skip_indexing_var.get():
                    print(f"\n🧠 Checking/Updating index for: {archive_dir}")
                    self._incremental_index(MODEL, archive_dir)
                else:
                    print(f"\n🧠 Using existing index (Read-Only) for: {archive_dir}")

                index_path = os.path.join(archive_dir, f"representations_{MODEL.lower()}.pkl")
                if os.path.exists(index_path):
                    try:
                        with open(index_path, 'rb') as f:
                            archive_df = pickle.load(f)
                            if "status" not in archive_df.columns: archive_df["status"] = "ok"
                            
                            # Filter for OK embeddings
                            filtered = archive_df[archive_df["status"] == "ok"][["identity", "embedding"]]
                            
                            # Add source dir column so we know where it came from
                            filtered["archive_dir"] = archive_dir 
                            
                            if not filtered.empty:
                                all_archive_embeddings.append(filtered)
                                print(f"   Loaded {len(filtered)} embeddings.")
                    except Exception as e:
                        print(f"⚠️ Failed to load index: {e}")

            if not all_archive_embeddings:
                print("❌ No valid embeddings found. Please run without 'Skip Indexing' once.")
                return

            # Concatenate all
            unified_df = pd.concat(all_archive_embeddings, ignore_index=True)
            archive_embeddings_np = np.array(unified_df['embedding'].tolist())
            
            # 2. Iterate References and Calculate Distances (Pure Math)
            for person, refs in REFERENCE_IMAGES_CONFIG.items():
                if self.stop_event.is_set(): break
                print(f"\n{'=' * 60}")
                print(f"🔍 Processing person: {person}")
                print(f"{'=' * 60}")

                potential_matches_df = pd.DataFrame()

                for ref in refs:
                    if self.stop_event.is_set(): break
                    if not os.path.exists(ref):
                        print(f"⚠️ Ref not found: {ref}")
                        continue

                    print(f"   Comparing against: {os.path.basename(ref)}")
                    try:
                        # Load ref image (with rotation fix AND RGB conversion)
                        img_input = self._load_image_fixed(ref)
                        
                        # Get Embedding
                        ref_reps = DeepFace.represent(
                            img_path=img_input,
                            model_name=MODEL,
                            detector_backend=DETECTOR,
                            enforce_detection=False,
                        )
                        if not ref_reps: continue
                        
                        ref_embedding = np.array(ref_reps[0]["embedding"])

                        # Math
                        if DIST_METRIC == "cosine":
                            ref_embedding_norm = ref_embedding / np.linalg.norm(ref_embedding)
                            archive_embeddings_norm = archive_embeddings_np / np.linalg.norm(archive_embeddings_np, axis=1, keepdims=True)
                            distances = 1 - np.dot(archive_embeddings_norm, ref_embedding_norm)
                        else:
                            distances = np.linalg.norm(archive_embeddings_np - ref_embedding, axis=1)

                        # Filter by Distance
                        matches_indices = np.where(distances <= MAX_DIST)[0]
                        
                        if len(matches_indices) > 0:
                            print(f"   ✓ Found {len(matches_indices)} matches within distance {MAX_DIST}")
                            
                            # Create temporary DF for these matches
                            matches = unified_df.iloc[matches_indices].copy()
                            matches["distance"] = distances[matches_indices]
                            potential_matches_df = pd.concat([potential_matches_df, matches], ignore_index=True)
                        else:
                            print(f"   ✓ No matches found")

                    except Exception as e:
                        print(f"⚠️ Error comparing {os.path.basename(ref)}: {e}")

                # 3. Copy Hits (Safe Copying)
                if not potential_matches_df.empty:
                    print(f"\n📁 Copying best matches for {person}...")
                    self.hits_log_df, copied = self._copy_hits_for_archive(potential_matches_df, person, "multi-source", self.hits_log_df)
                    total_copied += copied
                    if copied > 0:
                        print(f"✅ Copied {copied} new files.")
                    else:
                        print(f"ℹ️  Matches found, but already copied previously.")

            if not self.stop_event.is_set():
                print(f"\n{'=' * 60}")
                print(f"🎉 COMPLETE")
                print(f"{'=' * 60}")
                print(f"Total files copied: {total_copied}")

        except Exception as e:
            print(f"\n--- FATAL ERROR IN ANALYSIS ---: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            self.running_thread = None

    # --- Optimized Min Distance Calculation ---

    def _calculate_min_distances_optimized(self):
        if not self._get_current_config():
            return
        self.min_dist_display.delete(1.0, tk.END)
        print("\n--- Calculating Minimum Distances (Optimized) ---")
        
        all_archive_embeddings = [] 
        for archive_dir in ARCHIVE_DIRS:
            index_path = os.path.join(archive_dir, f"representations_{MODEL.lower()}.pkl")
            if os.path.exists(index_path):
                try:
                    with open(index_path, 'rb') as f:
                        archive_df = pickle.load(f)
                        if "status" not in archive_df.columns: archive_df["status"] = "ok" 
                        filtered = archive_df[archive_df["status"] == "ok"][["identity", "embedding"]]
                        if not filtered.empty:
                            all_archive_embeddings.append(filtered)
                except Exception as e:
                    print(f"⚠️ Could not load embeddings: {e}")
        
        if not all_archive_embeddings:
            messagebox.showerror("Error", "No embeddings indexes found. Please run 'Run Analysis' first.")
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
                    img_input = self._load_image_fixed(ref_path)
                    ref_reps = DeepFace.represent(img_path=img_input, model_name=MODEL, detector_backend=DETECTOR, enforce_detection=False)
                    if not ref_reps:
                        self.min_dist_display.insert(tk.END, f"Person '{person}': No face detected in ref.\n\n")
                        continue
                    ref_embedding = np.array(ref_reps[0]["embedding"])

                    if DIST_METRIC == "cosine":
                        ref_embedding_norm = ref_embedding / np.linalg.norm(ref_embedding)
                        archive_embeddings_norm = archive_embeddings_np / np.linalg.norm(archive_embeddings_np, axis=1, keepdims=True)
                        distances = 1 - np.dot(archive_embeddings_norm, ref_embedding_norm)
                    else:
                        distances = np.linalg.norm(archive_embeddings_np - ref_embedding, axis=1)
                    
                    min_index = np.argmin(distances)
                    lowest_dist_for_ref = distances[min_index]
                    closest_match_path = unified_embeddings_df.iloc[min_index]["identity"]
                    closest_match_path = os.path.abspath(closest_match_path)

                    self.min_dist_display.insert(tk.END, f"Ref: {os.path.basename(ref_path)} ({person})\n")
                    self.min_dist_display.insert(tk.END, f"  Dist: {lowest_dist_for_ref:.4f}\n")
                    self.min_dist_display.insert(tk.END, f"  Path: {closest_match_path}\n")
                    unique_tag = f"link_{int(time.time()*1000)}_{min_index}"
                    self.min_dist_display.insert(tk.END, "  👉 [ OPEN IMAGE ]\n\n", (unique_tag, "hyperlink"))
                    self.min_dist_display.tag_bind(unique_tag, "<Button-1>", lambda e, p=closest_match_path: self._open_file_from_link(p))

                except Exception as e:
                    print(f"Error: {e}")
                    self.min_dist_display.insert(tk.END, f"Error processing {person}: {e}\n\n")
        print("Minimum distances calculation complete.")
        self.min_dist_display.see(tk.END)

    # --- Utility Methods ---

    def _load_hits_log(self):
        if os.path.exists(self.HITS_LOG):
            try:
                df = pd.read_csv(self.HITS_LOG)
                return df
            except: pass
        return pd.DataFrame(columns=["person", "archive_dir", "identity", "distance", "copied_path"])

    def _atomic_pickle_save(self, df, path):
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, 'wb') as f:
                pickle.dump(df, f)
            os.replace(tmp_path, path)
        except Exception as e:
            print(f"⚠️ Failed to save checkpoint: {e}")
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass

    def _incremental_index(self, model_name, db_path):
        index_path = os.path.join(db_path, f"representations_{model_name.lower()}.pkl")
        bak_path = index_path + ".bak"

        if os.path.exists(index_path):
            try: shutil.copy2(index_path, bak_path)
            except: pass

        existing = pd.DataFrame()
        processed_paths = set()
        
        files_to_try = [index_path, index_path + ".tmp", bak_path]
        loaded = False
        for fp in files_to_try:
            if os.path.exists(fp):
                try:
                    with open(fp, 'rb') as f: existing = pickle.load(f)
                    if not existing.empty:
                        loaded = True
                        # Pruning only happens if we are NOT skipping index
                        if 'identity' in existing.columns:
                            processed_paths = set(existing['identity'].tolist())
                        break 
                except: pass
        
        if not existing.empty:
            if "status" not in existing.columns: existing["status"] = "ok"
            if "error" not in existing.columns: existing["error"] = None

        df = existing.copy()
        initial_df_len = len(df)
        last_save_time = time.time()

        def maybe_checkpoint(force=False):
            nonlocal last_save_time, df
            now = time.time()
            if force or (now - last_save_time) >= CHECKPOINT_INTERVAL_SECONDS:
                self._atomic_pickle_save(df, index_path)
                last_save_time = now
                print(f"💾 checkpoint saved ({len(df)} entries)")

        for root, _, files in os.walk(db_path):
            for file in files:
                if self.stop_event.is_set(): return df
                if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')): continue
                img_path = os.path.join(root, file)
                if img_path in processed_paths: continue

                try:
                    reps = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=False, detector_backend=DETECTOR)
                    if isinstance(reps, list) and len(reps) > 0:
                        emb = reps[0]["embedding"]
                        new_row = {"identity": img_path, "embedding": emb, "status": "ok", "error": None}
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                        processed_paths.add(img_path)
                except Exception as e:
                    msg = str(e)
                    print(f"⚠️ Skipping {img_path}: {msg}")
                    new_row = {"identity": img_path, "embedding": None, "status": "failed", "error": msg}
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    processed_paths.add(img_path)
                maybe_checkpoint(force=False)

        if len(df) > initial_df_len or not existing.equals(df): maybe_checkpoint(force=True)
        print(f"✅ index complete: {len(df)} entries saved")
        return df

    def _copy_hits_for_archive(self, df, person, archive_dir, hits_log_df):
        if df is None or df.empty: return hits_log_df, 0
        df = df.sort_values(by="distance").drop_duplicates(subset=["identity"])
        
        # Filtering is handled in the math loop now, but safety check
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
            
            # Get source archive if we merged multiple
            src_arch = row.get("archive_dir", archive_dir)

            if p in already_copied: continue 

            # --- LAZY CHECK: Only check existence NOW ---
            try:
                if os.path.exists(p):
                    dest_path = os.path.join(person_dir, os.path.basename(p))
                    shutil.copy2(p, dest_path)
                    copied += 1
                    new_rows.append({"person": person, "archive_dir": src_arch, "identity": p, "distance": dist, "copied_path": dest_path})
                    print(f"✅ Copied: {os.path.basename(p)} (dist: {dist:.4f})")
                else:
                    print(f"⚠️ Missing file (skipped copy): {p}")
            except Exception as e:
                print(f"⚠️ Copy failed for {p}: {e}")

        if new_rows:
            hits_log_df = pd.concat([hits_log_df, pd.DataFrame(new_rows)], ignore_index=True)
            hits_log_df.to_csv(self.HITS_LOG, index=False)
        return hits_log_df, copied

# ========== RUN THE GUI ==========
if __name__ == "__main__":
    app = FaceFinderGUI()
    app.mainloop()
