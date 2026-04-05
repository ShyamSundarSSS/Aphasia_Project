import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import queue
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from src.audio_recorder import AudioRecorder
from src.feature_extractor import FeatureExtractor
from src.model import MultimodalClassifier
from src.speech_to_text import SpeechToText
from src.scenario_generator import ScenarioGenerator
import sounddevice as sd
import soundfile as sf

class AphasiaTrainingGUI:
    """GUI for collecting speech samples and training the aphasia detection model."""
    
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("Aphasia Detection - Training Data Collection")
        self.root.geometry("900x700")
        
        # Initialize components
        self.recorder = AudioRecorder()
        self.transcriber = None  # Lazy load
        self.feature_extractor = None  # Lazy load
        self.scenario_generator = ScenarioGenerator()
        
        # State variables
        self.is_recording = False
        self.recording_duration = 30
        self.samples_collected = []
        self.training_data_dir = Path("training_data")
        self.training_data_dir.mkdir(exist_ok=True)
        
        # State variables for audio playback
        self.is_playing = False
        self.playback_thread = None
        
        # Queue for thread-safe GUI updates
        self.message_queue = queue.Queue()
        
        # Setup GUI
        self.setup_gui()
        
        # Load existing samples
        self.load_existing_samples()
        
        # Start queue processor
        self.process_queue()
    
    def setup_gui(self):
        """Setup the GUI layout."""
        # Title
        title_label = tk.Label(
            self.root, 
            text="Aphasia Detection - Training Data Collection", 
            font=("Arial", 16, "bold"),
            bg="#4CAF50",
            fg="white",
            pady=10
        )
        title_label.pack(fill=tk.X)
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Recording
        left_frame = ttk.LabelFrame(main_frame, text="Recording Controls", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Scenario selection
        ttk.Label(left_frame, text="Scenario:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.scenario_var = tk.StringVar()
        scenario_combo = ttk.Combobox(
            left_frame, 
            textvariable=self.scenario_var,
            values=self.scenario_generator.list_all_scenarios(),
            width=30,
            state="readonly"
        )
        scenario_combo.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        scenario_combo.current(0)
        
        ttk.Button(
            left_frame, 
            text="Random Scenario", 
            command=self.select_random_scenario
        ).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Scenario prompt display
        self.scenario_text = scrolledtext.ScrolledText(
            left_frame, 
            height=4, 
            width=40, 
            wrap=tk.WORD,
            font=("Arial", 9)
        )
        self.scenario_text.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Duration setting
        ttk.Label(left_frame, text="Duration (seconds):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.duration_var = tk.IntVar(value=30)
        duration_spinbox = ttk.Spinbox(
            left_frame, 
            from_=10, 
            to=120, 
            textvariable=self.duration_var,
            width=10
        )
        duration_spinbox.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Severity level selection
        ttk.Label(left_frame, text="True Severity Level:", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky=tk.W, pady=5)
        
        severity_frame = ttk.Frame(left_frame)
        severity_frame.grid(row=4, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.severity_var = tk.StringVar(value="Normal Speech")
        severity_combo = ttk.Combobox(
            severity_frame,
            textvariable=self.severity_var,
            values=["Normal Speech", "Mild Aphasia", "Moderate Aphasia", "Severe Aphasia", "Very Severe Aphasia"],
            width=28,
            state="readonly"
        )
        severity_combo.pack(side=tk.LEFT)
        
        # Help button
        help_btn = ttk.Button(
            severity_frame,
            text="?",
            width=2,
            command=self.show_severity_help
        )
        help_btn.pack(side=tk.LEFT, padx=5)
        
        # Aphasia type selection
        ttk.Label(left_frame, text="Aphasia Type (if applicable):").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.aphasia_type_var = tk.StringVar(value="None")
        aphasia_type_combo = ttk.Combobox(
            left_frame,
            textvariable=self.aphasia_type_var,
            values=["None", "Broca's", "Wernicke's", "Global", "Conduction", "Anomic", "Other"],
            width=30,
            state="readonly"
        )
        aphasia_type_combo.grid(row=5, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Recording buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=10)
        
        self.record_btn = ttk.Button(
            button_frame,
            text="🎤 Start Recording",
            command=self.start_recording,
            width=20
        )
        self.record_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = ttk.Button(
            button_frame,
            text="⏹️ Stop",
            command=self.stop_recording,
            state=tk.DISABLED,
            width=15
        )
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        # Status
        self.status_label = tk.Label(
            left_frame, 
            text="Ready to record", 
            font=("Arial", 10),
            fg="green"
        )
        self.status_label.grid(row=7, column=0, columnspan=3, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(left_frame, mode='indeterminate')
        self.progress.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Right panel - Sample list
        right_frame = ttk.LabelFrame(main_frame, text="Collected Samples", padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Sample count
        self.sample_count_label = tk.Label(
            right_frame,
            text="Total Samples: 0",
            font=("Arial", 10, "bold")
        )
        self.sample_count_label.pack(anchor=tk.W, pady=5)
        
        # Sample list
        list_frame = ttk.Frame(right_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.sample_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=("Courier", 9),
            height=20
        )
        self.sample_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.sample_listbox.yview)
        
        # Sample management buttons
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.play_btn = ttk.Button(
            btn_frame, 
            text="🔊 Play", 
            command=self.play_sample_audio,
            width=10
        )
        self.play_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_play_btn = ttk.Button(
            btn_frame,
            text="⏹️ Stop",
            command=self.stop_audio_playback,
            state=tk.DISABLED,
            width=10
        )
        self.stop_play_btn.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(btn_frame, text="🗑️ Delete", command=self.delete_sample, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📝 Details", command=self.view_sample_details, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="💾 Export", command=self.export_samples, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🗑️ Clear All", command=self.clear_samples, width=10).pack(side=tk.LEFT, padx=2)
        
        # Bottom panel - Training
        bottom_frame = ttk.LabelFrame(main_frame, text="Model Training", padding="10")
        bottom_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        train_info = ttk.Label(
            bottom_frame,
            text="Collect at least 50 samples (10+ per severity level) before training",
            font=("Arial", 9, "italic")
        )
        train_info.grid(row=0, column=0, columnspan=3, pady=5)
        
        ttk.Button(
            bottom_frame,
            text="🚀 Train Models",
            command=self.train_models,
            width=20
        ).grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Button(
            bottom_frame,
            text="💾 Save Current Model",
            command=self.save_model,
            width=20
        ).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(
            bottom_frame,
            text="📂 Load Model",
            command=self.load_model,
            width=20
        ).grid(row=1, column=2, padx=5, pady=5)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Update scenario display
        self.update_scenario_display()
    
    def select_random_scenario(self):
        """Select a random scenario."""
        scenario = self.scenario_generator.get_random_scenario()
        self.scenario_var.set(scenario['title'])
        self.update_scenario_display()
    
    def update_scenario_display(self):
        """Update the scenario prompt display."""
        scenario_title = self.scenario_var.get()
        scenario = self.scenario_generator.get_scenario_by_title(scenario_title)
        if scenario:
            self.current_scenario = scenario
            self.scenario_text.delete(1.0, tk.END)
            self.scenario_text.insert(1.0, scenario['prompt'])
    
    def start_recording(self):
        """Start recording audio."""
        if self.is_recording:
            return
        
        self.update_scenario_display()
        
        # Confirm ready
        response = messagebox.askyesno(
            "Ready to Record",
            f"Scenario: {self.current_scenario['title']}\n\n"
            f"{self.current_scenario['prompt']}\n\n"
            f"Duration: {self.duration_var.get()} seconds\n\n"
            "Press 'Yes' when ready to start recording."
        )
        
        if not response:
            return
        
        self.is_recording = True
        self.record_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress.start()
        
        # Start recording in separate thread
        thread = threading.Thread(target=self.record_audio)
        thread.daemon = True
        thread.start()
    
    def stop_recording(self):
        """Stop recording (not implemented for continuous recording)."""
        pass
    
    def record_audio(self):
        """Record audio in background thread."""
        try:
            self.message_queue.put(("status", "Recording... Speak now!", "red"))
            
            duration = self.duration_var.get()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sample_{timestamp}.wav"
            filepath = self.training_data_dir / filename
            
            # Record
            audio, audio_path = self.recorder.record_and_save(duration, str(filepath))
            
            self.message_queue.put(("status", "Processing audio...", "orange"))
            
            # Transcribe
            if self.transcriber is None:
                self.message_queue.put(("status", "Loading Whisper model...", "orange"))
                self.transcriber = SpeechToText(model_size="base")
            
            transcript_data = self.transcriber.get_annotated_transcript(str(filepath))
            
            # Extract features
            if self.feature_extractor is None:
                self.message_queue.put(("status", "Loading feature extractor...", "orange"))
                self.feature_extractor = FeatureExtractor()
            
            import librosa
            audio_array, sr = librosa.load(str(filepath), sr=16000)
            temporal_features = self.recorder.calculate_temporal_features(audio_array)
            
            all_features = self.feature_extractor.extract_all_features(
                str(filepath),
                transcript_data['plain_text'],
                temporal_features,
                self.current_scenario['keywords'] if hasattr(self, 'current_scenario') else None,
                self.current_scenario['prompt'] if hasattr(self, 'current_scenario') else None
            )
            
            # Create sample record
            severity_map = {
                "Normal Speech": 0,
                "Mild Aphasia": 1,
                "Moderate Aphasia": 2,
                "Severe Aphasia": 3,
                "Very Severe Aphasia": 4
            }
            
            sample = {
                'timestamp': timestamp,
                'filename': filename,
                'filepath': str(filepath),
                'scenario': self.current_scenario['title'] if hasattr(self, 'current_scenario') else "Unknown",
                'duration': duration,
                'severity_label': self.severity_var.get(),
                'severity_level': severity_map[self.severity_var.get()],
                'aphasia_type': self.aphasia_type_var.get(),
                'transcript': transcript_data['plain_text'],
                'word_count': all_features['linguistic']['word_count'],
                'features': {
                    'temporal': temporal_features,
                    'linguistic': all_features['linguistic'],
                    'grammar': all_features['grammar'],
                    'prosodic': all_features['prosodic'].tolist() if isinstance(all_features['prosodic'], np.ndarray) else all_features['prosodic']
                }
            }
            
            self.samples_collected.append(sample)
            self.save_sample_metadata(sample)
            
            self.message_queue.put(("sample_added", sample, None))
            self.message_queue.put(("status", "Sample saved successfully!", "green"))
            
        except Exception as e:
            self.message_queue.put(("error", f"Recording failed: {str(e)}", "red"))
        
        finally:
            self.is_recording = False
            self.message_queue.put(("recording_done", None, None))
    
    def play_sample_audio(self):
        """Play the audio of selected sample."""
        selection = self.sample_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a sample to play.")
            return
        
        if self.is_playing:
            messagebox.showinfo("Already Playing", "Audio is already playing. Stop it first.")
            return
        
        index = selection[0]
        sample = self.samples_collected[index]
        filepath = Path(sample['filepath'])
        
        if not filepath.exists():
            messagebox.showerror("File Not Found", f"Audio file not found:\n{filepath}")
            return
        
        self.is_playing = True
        self.play_btn.config(state=tk.DISABLED)
        self.stop_play_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"Playing: {sample['filename']}", fg="blue")
        
        self.playback_thread = threading.Thread(target=self._play_audio_thread, args=(filepath,))
        self.playback_thread.daemon = True
        self.playback_thread.start()
    
    def _play_audio_thread(self, filepath: Path):
        """Play audio in background thread."""
        try:
            audio_data, sample_rate = sf.read(str(filepath))
            sd.play(audio_data, sample_rate)
            sd.wait()
            self.message_queue.put(("playback_done", None, None))
        except Exception as e:
            self.message_queue.put(("error", f"Playback error: {str(e)}", "red"))
    
    def stop_audio_playback(self):
        """Stop audio playback."""
        if self.is_playing:
            sd.stop()
            self.is_playing = False
            self.play_btn.config(state=tk.NORMAL)
            self.stop_play_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Playback stopped", fg="orange")
    
    def save_sample_metadata(self, sample):
        """Save sample metadata to JSON file."""
        metadata_file = self.training_data_dir / "samples_metadata.json"
        
        try:
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = []
            
            metadata.append(sample)
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def load_existing_samples(self):
        """Load existing samples from metadata file."""
        metadata_file = self.training_data_dir / "samples_metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.samples_collected = json.load(f)
                
                self.update_sample_list()
            except Exception as e:
                print(f"Error loading samples: {e}")
    
    def update_sample_list(self):
        """Update the sample listbox."""
        self.sample_listbox.delete(0, tk.END)
        
        for i, sample in enumerate(self.samples_collected):
            display_text = f"{i+1}. {sample['timestamp']} | {sample['severity_label']} | {sample['scenario'][:20]}"
            self.sample_listbox.insert(tk.END, display_text)
        
        self.sample_count_label.config(text=f"Total Samples: {len(self.samples_collected)}")
    
    def delete_sample(self):
        """Delete selected sample."""
        selection = self.sample_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a sample to delete.")
            return
        
        index = selection[0]
        sample = self.samples_collected[index]
        
        response = messagebox.askyesno(
            "Confirm Delete",
            f"Delete this sample?\n\n"
            f"Timestamp: {sample['timestamp']}\n"
            f"Severity: {sample['severity_label']}\n"
            f"This will delete the audio file and cannot be undone!"
        )
        
        if not response:
            return
        
        try:
            filepath = Path(sample['filepath'])
            if filepath.exists():
                filepath.unlink()
        except Exception as e:
            print(f"Error deleting file: {e}")
        
        del self.samples_collected[index]
        self.save_all_metadata()
        self.update_sample_list()
        messagebox.showinfo("Success", "Sample deleted successfully!")
    
    def clear_samples(self):
        """Clear all samples."""
        if not self.samples_collected:
            messagebox.showinfo("No Samples", "No samples to clear.")
            return
        
        severity_counts = {}
        for sample in self.samples_collected:
            level = sample['severity_label']
            severity_counts[level] = severity_counts.get(level, 0) + 1
        
        msg = f"Delete ALL {len(self.samples_collected)} samples?\n\n"
        msg += "Distribution:\n"
        for level, count in sorted(severity_counts.items()):
            msg += f"  • {level}: {count} samples\n"
        msg += f"\nThis will delete all audio files and cannot be undone!"
        
        if not messagebox.askyesno("Confirm Clear All", msg):
            return
        
        deleted_count = 0
        for sample in self.samples_collected:
            try:
                filepath = Path(sample['filepath'])
                if filepath.exists():
                    filepath.unlink()
                    deleted_count += 1
            except Exception as e:
                print(f"Error deleting file: {e}")
        
        self.samples_collected = []
        self.save_all_metadata()
        self.update_sample_list()
        messagebox.showinfo("Clear Complete", f"Cleared all samples!\n\nDeleted: {deleted_count} files")
    
    def save_all_metadata(self):
        """Save all metadata."""
        metadata_file = self.training_data_dir / "samples_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.samples_collected, f, indent=2)
    
    def export_samples(self):
        """Export samples to a chosen directory."""
        if not self.samples_collected:
            messagebox.showinfo("No Samples", "No samples to export.")
            return
        
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return
        
        try:
            export_path = Path(export_dir) / f"aphasia_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            export_path.mkdir(parents=True, exist_ok=True)
            
            exported_count = 0
            for sample in self.samples_collected:
                src = Path(sample['filepath'])
                if src.exists():
                    dst = export_path / src.name
                    import shutil
                    shutil.copy2(src, dst)
                    exported_count += 1
            
            with open(export_path / "samples_metadata.json", 'w') as f:
                json.dump(self.samples_collected, f, indent=2)
            
            messagebox.showinfo("Success", f"Exported {exported_count} samples to:\n{export_path}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def view_sample_details(self):
        """View detailed information about selected sample."""
        selection = self.sample_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a sample to view.")
            return
        
        index = selection[0]
        sample = self.samples_collected[index]
        
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"Sample Details - {sample['timestamp']}")
        detail_window.geometry("600x650")
        
        text_widget = scrolledtext.ScrolledText(detail_window, wrap=tk.WORD, font=("Courier", 9))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        details = f"""
SAMPLE DETAILS
{'='*60}

Timestamp: {sample['timestamp']}
Filename: {sample['filename']}
File Path: {sample['filepath']}

CLASSIFICATION
{'='*60}
Severity Level: {sample['severity_label']} (Level {sample['severity_level']})
Aphasia Type: {sample.get('aphasia_type', 'Not specified')}
Scenario: {sample.get('scenario', 'Unknown')}
Duration: {sample['duration']} seconds

TRANSCRIPT
{'='*60}
{sample['transcript']}

SPEECH STATISTICS
{'='*60}
Total Words: {sample.get('word_count', 'N/A')}
"""
        
        if 'features' in sample and 'temporal' in sample['features']:
            temporal = sample['features']['temporal']
            details += f"""
TEMPORAL FEATURES
{'='*60}
Speech Segments: {temporal.get('num_segments', 'N/A')}
Average Pause: {temporal.get('avg_pause_duration', 0):.2f}s
Max Pause: {temporal.get('max_pause', 0):.2f}s
Speech Rate: {temporal.get('speech_rate', 0):.2f} seg/sec
Pause Ratio: {temporal.get('pause_ratio', 0):.2%}
"""
        
        text_widget.insert(1.0, details)
        text_widget.config(state=tk.DISABLED)
        
        button_frame = ttk.Frame(detail_window)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="🔊 Play Audio", command=lambda: self.play_audio_from_detail(sample['filepath'])).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=detail_window.destroy).pack(side=tk.LEFT, padx=5)
    
    def play_audio_from_detail(self, filepath: str):
        """Play audio from detail window."""
        if self.is_playing:
            messagebox.showinfo("Already Playing", "Audio is already playing.")
            return
        
        filepath = Path(filepath)
        if not filepath.exists():
            messagebox.showerror("File Not Found", f"Audio file not found:\n{filepath}")
            return
        
        self.is_playing = True
        thread = threading.Thread(target=self._play_audio_detail_thread, args=(filepath,))
        thread.daemon = True
        thread.start()
    
    def _play_audio_detail_thread(self, filepath: Path):
        """Play audio from detail window in background thread."""
        try:
            audio_data, sample_rate = sf.read(str(filepath))
            sd.play(audio_data, sample_rate)
            sd.wait()
            self.is_playing = False
        except Exception as e:
            messagebox.showerror("Playback Error", f"Could not play audio:\n{str(e)}")
            self.is_playing = False
    
    def show_severity_help(self):
        """Show help dialog explaining true severity level."""
        help_text = """TRUE SEVERITY LEVEL - What is this?

This is the CORRECT diagnosis that YOU (the clinician/researcher) 
determine based on clinical assessment.

📋 Why do we need this?
The machine learning model learns from examples. It needs to know 
the "correct answer" for each speech sample so it can learn to 
recognize patterns.

🎯 How to determine it:
Based on your clinical evaluation, standardized tests, or the 
patient's known diagnosis:

• Normal Speech - Healthy control, no language impairment
• Mild Aphasia - Slight word-finding issues, mostly intact
• Moderate Aphasia - Noticeable difficulties, broken grammar
• Severe Aphasia - Very limited output, major impairment
• Very Severe - Minimal speech, jargon, or near-complete impairment

⚠️ Important:
- This is NOT what the computer thinks (that comes after training)
- This is what YOU know to be clinically correct
- Accurate labeling = Better model performance

Example:
If you're recording a patient diagnosed with Moderate Broca's 
Aphasia, select "Moderate Aphasia" as the true severity level."""
        messagebox.showinfo("True Severity Level - Help", help_text)
    
    def train_models(self):
        """Train the classification models."""
        if len(self.samples_collected) < 50:
            messagebox.showwarning(
                "Insufficient Data",
                f"You have {len(self.samples_collected)} samples.\n"
                "Collect at least 50 samples for better model performance."
            )
            if not messagebox.askyesno("Continue?", "Train anyway?"):
                return
        
        severity_counts = {}
        for sample in self.samples_collected:
            level = sample['severity_level']
            severity_counts[level] = severity_counts.get(level, 0) + 1
        
        messagebox.showinfo(
            "Training Started",
            f"Training with {len(self.samples_collected)} samples\n\n"
            f"Distribution:\n" +
            "\n".join([f"Level {k}: {v} samples" for k, v in sorted(severity_counts.items())])
        )
        
        thread = threading.Thread(target=self.train_models_background)
        thread.daemon = True
        thread.start()
    
    def train_models_background(self):
        """Train models in background thread."""
        try:
            self.message_queue.put(("status", "Preparing training data...", "orange"))
            
            X_train = []
            y_train = []
            
            if self.feature_extractor is None:
                self.feature_extractor = FeatureExtractor()
            
            for sample in self.samples_collected:
                features = sample['features']
                feature_vector = self.extract_feature_vector(features)
                X_train.append(feature_vector)
                y_train.append(sample['severity_level'])
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            self.message_queue.put(("status", "Training LightGBM and XGBoost models...", "orange"))
            
            classifier = MultimodalClassifier()
            classifier.build_models()
            classifier.train_tree_models(X_train, y_train)
            
            model_dir = Path("trained_models") / datetime.now().strftime("%Y%m%d_%H%M%S")
            classifier.save_models(str(model_dir))
            
            self.message_queue.put(("status", f"Training complete! Model saved to {model_dir}", "green"))
            self.message_queue.put(("training_done", str(model_dir), None))
            
        except Exception as e:
            self.message_queue.put(("error", f"Training failed: {str(e)}", "red"))
    
    def extract_feature_vector(self, features):
        """Extract feature vector from features dict."""
        temporal = features['temporal']
        linguistic = features['linguistic']
        grammar = features['grammar']
        prosodic = features['prosodic']
        
        vector = [
            temporal['num_segments'],
            temporal['avg_segment_duration'],
            temporal['avg_pause_duration'],
            temporal['speech_rate'],
            temporal['pause_ratio'],
            temporal['max_pause'],
            temporal['segment_variability'],
            temporal['speaking_time'],
            temporal['total_time'],
            linguistic['word_count'],
            linguistic['unique_word_ratio'],
            linguistic['avg_word_length'],
            linguistic['filler_word_ratio'],
            linguistic['repetition_ratio'],
            linguistic.get('immediate_repetition_ratio', 0),
            linguistic['sentence_count'],
            linguistic['has_meaning'],
            linguistic['content_word_ratio'],
            linguistic.get('semantic_coherence', 0),
            grammar['grammar_errors'],
            grammar['errors_per_word'],
            1.0 if grammar['has_complete_sentences'] else 0.0,
            grammar['missing_function_words'],
            grammar['function_word_ratio'],
            prosodic[0], prosodic[1], prosodic[2], prosodic[3],
            prosodic[4], prosodic[5], prosodic[6],
            0.5, 0.5, 0.5, 0.5
        ]
        
        return vector
    
    def save_model(self):
        """Save current model."""
        messagebox.showinfo("Info", "Model saving is handled during training.\nCheck the 'trained_models' directory.")
    
    def load_model(self):
        """Load a trained model."""
        model_dir = filedialog.askdirectory(title="Select Model Directory")
        if model_dir:
            messagebox.showinfo("Info", f"Model loading functionality available in main prediction interface.\nSelected: {model_dir}")
    
    def process_queue(self):
        """Process messages from background threads."""
        try:
            while True:
                msg_type, msg_data, msg_color = self.message_queue.get_nowait()
                
                if msg_type == "status":
                    self.status_label.config(text=msg_data, fg=msg_color)
                
                elif msg_type == "sample_added":
                    self.update_sample_list()
                
                elif msg_type == "recording_done":
                    self.record_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.progress.stop()
                
                elif msg_type == "playback_done":
                    self.is_playing = False
                    self.play_btn.config(state=tk.NORMAL)
                    self.stop_play_btn.config(state=tk.DISABLED)
                    self.status_label.config(text="Playback completed", fg="green")
                
                elif msg_type == "error":
                    messagebox.showerror("Error", msg_data)
                    self.record_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.progress.stop()
                    if self.is_playing:
                        self.is_playing = False
                        self.play_btn.config(state=tk.NORMAL)
                        self.stop_play_btn.config(state=tk.DISABLED)
                
                elif msg_type == "training_done":
                    messagebox.showinfo("Success", f"Training completed!\nModel saved to:\n{msg_data}")
        
        except queue.Empty:
            pass
        
        self.root.after(100, self.process_queue)


def main():
    """Main entry point for GUI application."""
    root = tk.Tk()
    app = AphasiaTrainingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
