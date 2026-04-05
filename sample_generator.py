import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
import queue
from typing import List, Dict
import time
import re

# Text-to-speech engines
try:
    import pyttsx3
    TTS_ENGINE = "pyttsx3"
except ImportError:
    print("pyttsx3 not installed. Install with: pip install pyttsx3")
    TTS_ENGINE = None

class SampleGenerator:
    """Reads text from files and speaks them aloud for sample collection."""
    
    def __init__(self, root):
        """Initialize the sample generator GUI."""
        self.root = root
        self.root.title("Aphasia Sample Generator - Text-to-Speech")
        self.root.geometry("850x850")  # CHANGED: Made window taller
        
        # Initialize TTS engine
        self.tts_engine = None
        self.is_speaking = False
        self.current_text = ""
        self.message_queue = queue.Queue()
        self.available_voices = []
        
        # Sample configurations
        self.samples = []
        self.current_index = 0
        
        # NEW: Aphasia simulation settings
        self.simulate_aphasia = tk.BooleanVar(value=False)
        self.aphasia_type = tk.StringVar(value="None")
        
        # Initialize TTS
        self.init_tts()
        
        # Setup GUI
        self.setup_gui()
        
        # Load available voices
        self.load_voices()
        
        # Start queue processor
        self.process_queue()
        
        # NEW: Test audio output on startup
        self.test_audio_output()
    
    def init_tts(self):
        """Initialize text-to-speech engine."""
        if TTS_ENGINE == "pyttsx3":
            try:
                self.tts_engine = pyttsx3.init()
                
                # CHANGED: Increase default volume to maximum
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 1.0)  # Maximum volume
                
                # Try to get audio device
                try:
                    # Force audio output to default device
                    import platform
                    if platform.system() == "Windows":
                        # Windows-specific: ensure SAPI5 is being used
                        pass
                except:
                    pass
                
                print("TTS engine initialized successfully")
            except Exception as e:
                print(f"Error initializing TTS: {e}")
                self.tts_engine = None
                messagebox.showerror(
                    "TTS Error",
                    f"Failed to initialize text-to-speech:\n{str(e)}\n\n"
                    "Make sure your audio drivers are working."
                )
        else:
            messagebox.showerror(
                "TTS Not Available",
                "Text-to-speech engine not available.\n"
                "Install pyttsx3: pip install pyttsx3"
            )
    
    def test_audio_output(self):
        """Test if audio output is working."""
        if not self.tts_engine:
            return
        
        print("\n" + "="*50)
        print("TESTING AUDIO OUTPUT")
        print("="*50)
        print("You should hear: 'Audio test successful'")
        print("If you don't hear anything:")
        print("1. Check your speaker volume")
        print("2. Check Windows sound settings")
        print("3. Make sure speakers are plugged in")
        print("="*50 + "\n")
        
        try:
            self.tts_engine.say("Audio test successful")
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Audio test failed: {e}")
            messagebox.showwarning(
                "Audio Test Failed",
                f"Could not play audio test:\n{str(e)}\n\n"
                "Check your audio settings and speakers."
            )
    
    def load_voices(self):
        """Load available voices from TTS engine."""
        if self.tts_engine:
            try:
                voices = self.tts_engine.getProperty('voices')
                self.available_voices = voices
                
                # Populate voice dropdown
                voice_names = []
                for i, voice in enumerate(voices):
                    # Extract a readable name
                    name = voice.name
                    # Try to determine gender from voice properties
                    if 'female' in name.lower() or 'zira' in name.lower() or 'hazel' in name.lower():
                        gender = "Female"
                    elif 'male' in name.lower() or 'david' in name.lower() or 'mark' in name.lower():
                        gender = "Male"
                    else:
                        gender = "Unknown"
                    
                    display_name = f"{gender} - {name[:40]}"
                    voice_names.append(display_name)
                
                self.voice_combo['values'] = voice_names
                if voice_names:
                    self.voice_combo.current(0)
                    
            except Exception as e:
                print(f"Error loading voices: {e}")
    
    def setup_gui(self):
        """Setup the GUI layout."""
        # Title
        title_label = tk.Label(
            self.root,
            text="Aphasia Sample Generator - Text-to-Speech",
            font=("Arial", 14, "bold"),
            bg="#2196F3",
            fg="white",
            pady=10
        )
        title_label.pack(fill=tk.X)
        
        # CHANGED: Create canvas with scrollbar for main content
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # CHANGED: Use scrollable_frame instead of main_frame
        main_frame = ttk.Frame(scrollable_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="Load text samples to simulate aphasia speech patterns.\n"
                 "The system will read them aloud for recording by the training GUI.",
            font=("Arial", 9),
            justify=tk.CENTER
        )
        instructions.pack(pady=5)
        
        # File operations frame
        file_frame = ttk.LabelFrame(main_frame, text="Load Samples", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            file_frame,
            text="📁 Load Text File",
            command=self.load_text_file,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            file_frame,
            text="📂 Load Multiple Files",
            command=self.load_multiple_files,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            file_frame,
            text="📋 Load Sample Library",
            command=self.load_sample_library,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        # Sample list frame - CHANGED: Reduced height
        list_frame = ttk.LabelFrame(main_frame, text="Loaded Samples", padding="10")
        list_frame.pack(fill=tk.X, pady=5)  # CHANGED: fill=tk.X instead of BOTH
        
        # Sample count
        self.sample_count_label = tk.Label(
            list_frame,
            text="Samples: 0",
            font=("Arial", 10, "bold")
        )
        self.sample_count_label.pack(anchor=tk.W, pady=5)
        
        # Listbox with scrollbar
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        list_scrollbar = ttk.Scrollbar(list_container)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.sample_listbox = tk.Listbox(
            list_container,
            yscrollcommand=list_scrollbar.set,
            font=("Courier", 9),
            height=4  # CHANGED: Reduced from 6 to 4
        )
        self.sample_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.config(command=self.sample_listbox.yview)
        
        self.sample_listbox.bind('<<ListboxSelect>>', self.on_sample_select)
        
        # Text preview frame - CHANGED: Reduced height
        preview_frame = ttk.LabelFrame(main_frame, text="Current Sample Text", padding="10")
        preview_frame.pack(fill=tk.X, pady=5)  # CHANGED: fill=tk.X
        
        self.text_display = scrolledtext.ScrolledText(
            preview_frame,
            height=3,  # CHANGED: Reduced from 4 to 3
            wrap=tk.WORD,
            font=("Arial", 10)
        )
        self.text_display.pack(fill=tk.BOTH, expand=True)
        
        # Voice settings frame
        voice_frame = ttk.LabelFrame(main_frame, text="Voice Settings", padding="10")
        voice_frame.pack(fill=tk.X, pady=5)
        
        # Voice selection
        ttk.Label(voice_frame, text="Voice:", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.voice_var = tk.StringVar()
        self.voice_combo = ttk.Combobox(
            voice_frame,
            textvariable=self.voice_var,
            state="readonly",
            width=40
        )
        self.voice_combo.grid(row=0, column=1, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.voice_combo.bind('<<ComboboxSelected>>', self.change_voice)
        
        # Speed control
        ttk.Label(voice_frame, text="Speed (WPM):", font=("Arial", 9)).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.speed_var = tk.IntVar(value=150)
        self.speed_label = tk.Label(voice_frame, text="150", font=("Arial", 9, "bold"), width=5)
        self.speed_label.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        speed_scale = ttk.Scale(
            voice_frame,
            from_=50,
            to=300,
            variable=self.speed_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.update_speed_label
        )
        speed_scale.grid(row=1, column=2, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Volume control
        ttk.Label(voice_frame, text="Volume:", font=("Arial", 9)).grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.volume_var = tk.DoubleVar(value=1.0)  # CHANGED: Start at maximum
        self.volume_label = tk.Label(voice_frame, text="100%", font=("Arial", 9, "bold"), width=5)
        self.volume_label.grid(row=2, column=1, sticky=tk.W, padx=5)
        
        volume_scale = ttk.Scale(
            voice_frame,
            from_=0.0,
            to=1.0,
            variable=self.volume_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.update_volume_label
        )
        volume_scale.grid(row=2, column=2, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Preset buttons
        preset_frame = ttk.Frame(voice_frame)
        preset_frame.grid(row=3, column=0, columnspan=4, pady=5)
        
        ttk.Button(preset_frame, text="Very Slow", command=lambda: self.apply_preset(75, 1.0), width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Slow", command=lambda: self.apply_preset(100, 1.0), width=12).pack(side=tk.LEFT, padx=2)  # CHANGED: volume to 1.0
        ttk.Button(preset_frame, text="Normal", command=lambda: self.apply_preset(150, 1.0), width=12).pack(side=tk.LEFT, padx=2)  # CHANGED: volume to 1.0
        ttk.Button(preset_frame, text="Fast", command=lambda: self.apply_preset(200, 1.0), width=12).pack(side=tk.LEFT, padx=2)  # CHANGED: volume to 1.0
        ttk.Button(preset_frame, text="Test Voice", command=self.test_voice, width=12).pack(side=tk.LEFT, padx=2)
        
        # NEW: Audio troubleshooting button
        ttk.Button(preset_frame, text="🔧 Audio Help", command=self.show_audio_help, width=12).pack(side=tk.LEFT, padx=2)
        
        # NEW: Aphasia Simulation Frame
        sim_frame = ttk.LabelFrame(main_frame, text="Aphasia Simulation", padding="10")
        sim_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(
            sim_frame,
            text="Enable Aphasia Simulation",
            variable=self.simulate_aphasia,
            command=self.toggle_simulation
        ).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(sim_frame, text="Simulation Type:", font=("Arial", 9)).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        aphasia_combo = ttk.Combobox(
            sim_frame,
            textvariable=self.aphasia_type,
            values=["Auto-detect from text", "Broca's (Nonfluent)", "Wernicke's (Fluent)", "Severe", "Very Severe"],
            width=25,
            state="readonly"
        )
        aphasia_combo.grid(row=0, column=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        aphasia_combo.current(0)
        
        # Simulation info
        sim_info = ttk.Label(
            sim_frame,
            text="Simulation adds natural pauses, hesitations, and speech patterns matching aphasia severity",
            font=("Arial", 8, "italic"),
            foreground="gray"
        )
        sim_info.grid(row=1, column=0, columnspan=3, pady=5)
        
        # Control buttons frame - NOW VISIBLE!
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.speak_btn = ttk.Button(
            control_frame,
            text="🔊 Speak Current",
            command=self.speak_current,
            width=18
        )
        self.speak_btn.pack(side=tk.LEFT, padx=5)
        
        self.replay_btn = ttk.Button(
            control_frame,
            text="🔁 Replay",
            command=self.replay_current,
            width=15
        )
        self.replay_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(
            control_frame,
            text="⏹️ Stop",
            command=self.stop_speaking,
            state=tk.DISABLED,
            width=12
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="⏮️ Previous",
            command=self.previous_sample,
            width=12
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="⏭️ Next",
            command=self.next_sample,
            width=12
        ).pack(side=tk.LEFT, padx=5)
        
        # Sample management buttons
        manage_frame = ttk.Frame(main_frame)
        manage_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            manage_frame,
            text="🗑️ Delete Selected",
            command=self.delete_selected_sample,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            manage_frame,
            text="🗑️ Clear All",
            command=self.clear_all_samples,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            manage_frame,
            text="💾 Save to File",
            command=self.save_current_text,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_label = tk.Label(
            main_frame,
            text="Ready - Load text samples to begin",
            font=("Arial", 10),
            fg="green"
        )
        self.status_label.pack(pady=5)
        
        # ADDED: Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def toggle_simulation(self):
        """Toggle aphasia simulation and provide feedback."""
        if self.simulate_aphasia.get():
            self.status_label.config(
                text="✓ Aphasia simulation enabled - Speech will include natural hesitations",
                fg="blue"
            )
        else:
            self.status_label.config(text="Aphasia simulation disabled", fg="gray")
    
    def change_voice(self, event=None):
        """Change the TTS voice."""
        if self.tts_engine and self.available_voices:
            selected_index = self.voice_combo.current()
            if 0 <= selected_index < len(self.available_voices):
                voice = self.available_voices[selected_index]
                self.tts_engine.setProperty('voice', voice.id)
                self.status_label.config(text=f"Voice changed to: {voice.name[:30]}", fg="blue")
    
    def update_speed_label(self, value):
        """Update speed label and apply to TTS."""
        speed = int(float(value))
        self.speed_label.config(text=str(speed))
        if self.tts_engine:
            self.tts_engine.setProperty('rate', speed)
    
    def update_volume_label(self, value):
        """Update volume label and apply to TTS."""
        volume = float(value)
        percentage = int(volume * 100)
        self.volume_label.config(text=f"{percentage}%")
        
        if self.tts_engine:
            self.tts_engine.setProperty('volume', volume)
            print(f"Volume set to {percentage}%")  # Debug output
    
    def apply_preset(self, speed: int, volume: float):
        """Apply preset voice settings."""
        self.speed_var.set(speed)
        self.volume_var.set(volume)
        self.update_speed_label(speed)
        self.update_volume_label(volume)
        self.status_label.config(text=f"Preset applied: {speed} WPM, {int(volume*100)}% volume", fg="blue")
    
    def test_voice(self):
        """Test current voice settings with a short phrase."""
        if not self.tts_engine:
            messagebox.showerror("TTS Error", "Text-to-speech engine not available.")
            return
        
        if self.is_speaking:
            messagebox.showinfo("Busy", "Already speaking. Stop current playback first.")
            return
        
        # CHANGED: Make test more obvious
        test_text = "Testing. Can you hear me? Volume test. One, two, three."
        
        print("\n🔊 SPEAKING NOW - Turn up your speakers if you can't hear!")
        
        self.is_speaking = True
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="🔊 TESTING VOICE - Turn up speakers!", fg="orange")
        
        def test_thread():
            try:
                # Set volume to maximum for test
                current_vol = self.tts_engine.getProperty('volume')
                self.tts_engine.setProperty('volume', 1.0)
                
                self.tts_engine.say(test_text)
                self.tts_engine.runAndWait()
                
                # Restore volume
                self.tts_engine.setProperty('volume', current_vol)
                
                self.message_queue.put(("test_done", None))
            except Exception as e:
                self.message_queue.put(("error", f"Test error: {str(e)}"))
        
        thread = threading.Thread(target=test_thread)
        thread.daemon = True
        thread.start()
    
    def load_text_file(self):
        """Load a single text file."""
        filepath = filedialog.askopenfilename(
            title="Select Text File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                if text:
                    sample = {
                        'text': text,
                        'filename': Path(filepath).name,
                        'source': filepath
                    }
                    self.samples.append(sample)
                    self.update_sample_list()
                    messagebox.showinfo("Success", f"Loaded: {Path(filepath).name}")
                else:
                    messagebox.showwarning("Empty File", "The selected file is empty.")
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
    
    def load_multiple_files(self):
        """Load multiple text files."""
        filepaths = filedialog.askopenfilenames(
            title="Select Text Files",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if filepaths:
            loaded = 0
            for filepath in filepaths:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    if text:
                        sample = {
                            'text': text,
                            'filename': Path(filepath).name,
                            'source': filepath
                        }
                        self.samples.append(sample)
                        loaded += 1
                
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
            
            self.update_sample_list()
            messagebox.showinfo("Success", f"Loaded {loaded} samples")
    
    def load_sample_library(self):
        """Load pre-defined sample library for different aphasia types."""
        library_dir = Path("sample_texts")
        library_dir.mkdir(exist_ok=True)
        
        # Create sample files if they don't exist
        self.create_sample_library(library_dir)
        
        # Load all samples from library
        loaded = 0
        for txt_file in library_dir.glob("*.txt"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                if text:
                    sample = {
                        'text': text,
                        'filename': txt_file.name,
                        'source': str(txt_file)
                    }
                    self.samples.append(sample)
                    loaded += 1
            
            except Exception as e:
                print(f"Error loading {txt_file}: {e}")
        
        self.update_sample_list()
        messagebox.showinfo("Sample Library", f"Loaded {loaded} samples from library")
    
    def create_sample_library(self, library_dir: Path):
        """Create sample text files for different aphasia severities."""
        samples = {
            "normal_speech_1.txt": "Yesterday I went to the market with my friend. We bought vegetables and then had lunch at a restaurant. After coming home, I watched a movie and relaxed.",
            "mild_aphasia_1.txt": "Yesterday I... uh... went to the place, the big shop... market. I was with my friend. We buy... bought vegetables and then we ate at, um, food place... restaurant. Later I came home and watched... watched a movie.",
            "moderate_aphasia_1.txt": "Yesterday I go market... with friend. We buy vege... veg... the green things. Eat food at rest... restan... the eating place. Home later... watch movie. Good day.",
            "severe_aphasia_1.txt": "Yesterday... market friend. Buy... green... uh... vege. Eat... rest... reslan. Home... movie... good. Yes.",
            "very_severe_aphasia_1.txt": "Yest... mar... frin go. Buy gree... slapo. Eat res... slan. Home mova... yes good-good.",
            "normal_speech_2.txt": "My family and I had dinner together last night. We talked about our day and shared stories. The food was delicious and everyone enjoyed the meal.",
            "mild_aphasia_2.txt": "My family... we had dinner, um, together last night. We talked about... about our day and, uh, shared stories. The food was... was delicious and everyone, um, enjoyed the meal.",
            "moderate_aphasia_2.txt": "Family dinner... last night. Talk day... stories. Food... delicious... everyone enjoy... enjoyed meal. Good time.",
            "severe_aphasia_2.txt": "Family... dinner. Talk... day. Food good... everyone. Yes... good.",
            "broca_pattern_1.txt": "I... want... go... store. Need... buy... milk... bread. Tomorrow... maybe.",
            "wernicke_pattern_1.txt": "Well you know the thing about the place where the stuff goes and then the people are doing the things with the whatever and yes that's how it works with the situation.",
            "normal_speech_3.txt": "I enjoy reading books in my free time. My favorite genres are mystery and science fiction. Last week I finished a really interesting novel about time travel.",
        }
        
        for filename, content in samples.items():
            filepath = library_dir / filename
            if not filepath.exists():
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
    
    def update_sample_list(self):
        """Update the sample listbox."""
        self.sample_listbox.delete(0, tk.END)
        
        for i, sample in enumerate(self.samples):
            display = f"{i+1}. {sample['filename']}"
            self.sample_listbox.insert(tk.END, display)
        
        self.sample_count_label.config(text=f"Samples: {len(self.samples)}")
        
        # Select first item if available
        if self.samples and self.sample_listbox.size() > 0:
            self.sample_listbox.selection_set(0)
            self.on_sample_select(None)
    
    def on_sample_select(self, event):
        """Handle sample selection."""
        selection = self.sample_listbox.curselection()
        if selection:
            self.current_index = selection[0]
            sample = self.samples[self.current_index]
            
            # Display text
            self.text_display.delete(1.0, tk.END)
            self.text_display.insert(1.0, sample['text'])
            
            self.current_text = sample['text']
    
    def replay_current(self):
        """Replay the current sample."""
        if not self.current_text:
            messagebox.showwarning("No Text", "Please select a sample first.")
            return
        
        if self.is_speaking:
            messagebox.showinfo("Already Speaking", "Audio is already playing. Stop it first to replay.")
            return
        
        self.speak_current()
        self.status_label.config(text="Replaying... Record this in the training GUI!", fg="blue")
    
    def speak_current(self):
        """Speak the current sample text with optional aphasia simulation."""
        if not self.current_text:
            messagebox.showwarning("No Text", "Please select a sample first.")
            return
        
        if not self.tts_engine:
            messagebox.showerror("TTS Error", "Text-to-speech engine not available.")
            return
        
        if self.is_speaking:
            messagebox.showinfo("Already Speaking", "Audio is already playing.")
            return
        
        self.is_speaking = True
        self.speak_btn.config(state=tk.DISABLED)
        self.replay_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        if self.simulate_aphasia.get():
            self.status_label.config(text="Speaking with aphasia simulation... Record this!", fg="red")
        else:
            self.status_label.config(text="Speaking... Record this in the training GUI!", fg="red")
        
        # Speak in separate thread
        thread = threading.Thread(target=self._speak_thread)
        thread.daemon = True
        thread.start()
    
    def _speak_thread(self):
        """Speak text in background thread with optional aphasia simulation."""
        try:
            print(f"\n🔊 SPEAKING: {self.current_text[:50]}...")
            print("📢 If you can't hear, check:")
            print("   1. Computer volume (turn it UP!)")
            print("   2. Speaker/headphone connection")
            print("   3. Windows sound settings")
            
            if self.simulate_aphasia.get():
                # Simulate aphasia speech patterns
                self._speak_with_aphasia_simulation()
            else:
                # Normal speech - ensure volume is at maximum
                current_vol = self.tts_engine.getProperty('volume')
                self.tts_engine.setProperty('volume', 1.0)
                
                # FIXED: Use stop() before speaking to clear any pending speech
                self.tts_engine.stop()
                
                self.tts_engine.say(self.current_text)
                self.tts_engine.runAndWait()
                
                # FIXED: Explicitly stop after completion
                self.tts_engine.stop()
                
                # Restore volume
                self.tts_engine.setProperty('volume', current_vol)
            
            print("✓ Speech completed!")
            self.message_queue.put(("speaking_done", None))
        
        except Exception as e:
            print(f"❌ Speech error: {str(e)}")
            # FIXED: Always stop engine on error
            try:
                self.tts_engine.stop()
            except:
                pass
            self.message_queue.put(("error", f"Speech error: {str(e)}"))
        
        finally:
            # FIXED: Ensure is_speaking is reset
            self.is_speaking = False
    
    def _speak_with_aphasia_simulation(self):
        """Speak with simulated aphasia patterns including pauses and hesitations."""
        text = self.current_text
        
        # Detect aphasia type from text or use selected type
        aphasia_type = self._detect_aphasia_type(text)
        
        # ENHANCED: Adjust TTS settings based on aphasia type
        original_rate = self.tts_engine.getProperty('rate')
        original_volume = self.tts_engine.getProperty('volume')
        
        # CHANGED: Ensure volume is always at maximum during speech
        self.tts_engine.setProperty('volume', 1.0)
        
        # Set slower speed for aphasia simulation
        if "Severe" in aphasia_type or "Very Severe" in aphasia_type:
            self.tts_engine.setProperty('rate', max(50, original_rate - 80))
        elif "Nonfluent" in aphasia_type:
            self.tts_engine.setProperty('rate', max(70, original_rate - 60))
        else:
            self.tts_engine.setProperty('rate', max(100, original_rate - 30))
        
        # Parse text and add pauses
        segments = self._parse_text_for_simulation(text, aphasia_type)
        
        print(f"Speaking {len(segments)} segments with {aphasia_type} simulation...")
        
        # FIXED: Clear any pending speech before starting
        self.tts_engine.stop()
        
        try:
            # Speak each segment with appropriate timing
            for i, segment in enumerate(segments):
                # FIXED: Check if we should stop
                if not self.is_speaking:
                    print("Speech interrupted by user")
                    break
                
                if segment['type'] == 'text':
                    print(f"  Segment {i+1}: '{segment['content'][:30]}...'")
                    
                    # ENHANCED: For very severe, speak word by word with pauses
                    if "Very Severe" in aphasia_type or "Severe" in aphasia_type:
                        words = segment['content'].split()
                        for word in words:
                            if not self.is_speaking:
                                break
                            self.tts_engine.say(word)
                            self.tts_engine.runAndWait()
                            time.sleep(0.4 if "Very Severe" in aphasia_type else 0.25)
                    else:
                        self.tts_engine.say(segment['content'])
                        self.tts_engine.runAndWait()
                
                elif segment['type'] == 'pause':
                    print(f"  Pause: {segment['duration']:.1f}s")
                    # FIXED: Sleep in smaller chunks to allow interruption
                    pause_time = segment['duration']
                    elapsed = 0
                    while elapsed < pause_time and self.is_speaking:
                        sleep_time = min(0.1, pause_time - elapsed)
                        time.sleep(sleep_time)
                        elapsed += sleep_time
                
                elif segment['type'] == 'repeat':
                    word = segment['content']
                    print(f"  Repeat '{word}' x{segment['count']}")
                    for rep in range(segment['count']):
                        if not self.is_speaking:
                            break
                        self.tts_engine.say(word)
                        self.tts_engine.runAndWait()
                        time.sleep(0.2)
        
        finally:
            # FIXED: Always restore original settings and stop engine
            self.tts_engine.stop()
            self.tts_engine.setProperty('rate', original_rate)
            self.tts_engine.setProperty('volume', original_volume)
            print("✓ Aphasia simulation completed!")
    
    def _detect_aphasia_type(self, text: str) -> str:
        """Detect aphasia severity from text patterns."""
        simulation_type = self.aphasia_type.get()
        
        if simulation_type != "Auto-detect from text":
            return simulation_type
        
        # Auto-detect based on text characteristics
        text_lower = text.lower()
        word_count = len(text.split())
        
        # Check for ellipsis and pause markers
        ellipsis_count = text.count('...')
        dot_count = text.count('.') - ellipsis_count
        
        # Check for stuttering patterns
        has_stuttering = bool(re.search(r'\b(\w+)\.\.\. \1', text))
        
        # Check for very fragmented speech
        very_short_words = len([w for w in text.split() if len(w) <= 3])
        
        if ellipsis_count >= 5 or very_short_words > word_count * 0.6:
            return "Very Severe"
        elif ellipsis_count >= 3 or has_stuttering:
            return "Severe"
        elif ellipsis_count >= 1 or text.count('uh') > 0 or text.count('um') > 0:
            return "Broca's (Nonfluent)"
        elif word_count > 20 and dot_count < 2:
            return "Wernicke's (Fluent)"
        else:
            return "Mild"
    
    def _parse_text_for_simulation(self, text: str, aphasia_type: str) -> List[Dict]:
        """Parse text into speech segments with pauses based on aphasia type."""
        segments = []
        
        # Define pause durations based on severity
        pause_map = {
            "Broca's (Nonfluent)": {'short': 0.5, 'medium': 1.2, 'long': 2.0},
            "Wernicke's (Fluent)": {'short': 0.15, 'medium': 0.3, 'long': 0.5},
            "Severe": {'short': 0.8, 'medium': 1.8, 'long': 3.0},
            "Very Severe": {'short': 1.2, 'medium': 2.5, 'long': 4.0},
            "Mild": {'short': 0.3, 'medium': 0.7, 'long': 1.2}
        }
        
        pauses = pause_map.get(aphasia_type, pause_map["Mild"])
        
        # ENHANCED: Detect and handle repetitions (e.g., "watched... watched")
        # Pattern: word... word or word.. word
        text = re.sub(
            r'\b(\w+)\.\.\. \1\b',
            r'[REPEAT:\1:2] \1',
            text,
            flags=re.IGNORECASE
        )
        
        # Split text by various pause markers
        text = text.replace('...', ' [LONG] ')
        text = text.replace('..', ' [MEDIUM] ')
        
        # Don't add pauses for commas in Wernicke's (fluent)
        if "Fluent" not in aphasia_type:
            text = text.replace(',', ' [SHORT] ')
        
        # Add pauses after filler words
        text = re.sub(r'\b(uh|um|er|ah|uhh|umm|err|ahh)\b', r'\1 [SHORT]', text, flags=re.IGNORECASE)
        
        # ENHANCED: For severe aphasia, add pauses after each word
        if "Severe" in aphasia_type or "Very Severe" in aphasia_type:
            # Add SHORT pause markers between words
            text = re.sub(r'(\w+)\s+(?!\[)', r'\1 [SHORT] ', text)
        
        # Split by pause markers and repetition markers
        parts = re.split(r'\s*\[(SHORT|MEDIUM|LONG|REPEAT:\w+:\d+)\]\s*', text)
        
        for i, part in enumerate(parts):
            part = part.strip()
            
            if part in ['SHORT', 'MEDIUM', 'LONG']:
                # Add pause
                pause_duration = pauses[part.lower()]
                segments.append({
                    'type': 'pause',
                    'duration': pause_duration
                })
            elif part.startswith('REPEAT:'):
                # Handle repetition: REPEAT:word:count
                match = re.match(r'REPEAT:(\w+):(\d+)', part)
                if match:
                    word = match.group(1)
                    count = int(match.group(2))
                    segments.append({
                        'type': 'repeat',
                        'content': word,
                        'count': count
                    })
            elif part:
                # Add text segment
                # ENHANCED: Don't add large chunks for severe aphasia
                if "Very Severe" in aphasia_type or "Severe" in aphasia_type:
                    # Already handled word-by-word in _speak_with_aphasia_simulation
                    segments.append({
                        'type': 'text',
                        'content': part
                    })
                else:
                    segments.append({
                        'type': 'text',
                        'content': part
                    })
        
        return segments
    
    def stop_speaking(self):
        """Stop speaking."""
        print("🛑 Stopping speech...")
        
        # Set flag first
        self.is_speaking = False
        
        # Stop the engine
        if self.tts_engine:
            try:
                self.tts_engine.stop()
                print("✓ TTS engine stopped")
            except Exception as e:
                print(f"Error stopping engine: {e}")
        
        # Update UI
        self.speak_btn.config(state=tk.NORMAL)
        self.replay_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Stopped", fg="orange")
    
    def delete_selected_sample(self):
        """Delete the currently selected sample."""
        selection = self.sample_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a sample to delete.")
            return
        
        index = selection[0]
        sample = self.samples[index]
        
        response = messagebox.askyesno(
            "Confirm Delete",
            f"Delete sample: {sample['filename']}?\n\nThis cannot be undone."
        )
        
        if response:
            # Delete from list
            del self.samples[index]
            
            # Update display
            self.update_sample_list()
            
            # Clear text if it was the deleted one
            if self.current_index == index:
                self.text_display.delete(1.0, tk.END)
                self.current_text = ""
                self.current_index = 0
            
            self.status_label.config(text=f"Deleted: {sample['filename']}", fg="orange")
    
    def clear_all_samples(self):
        """Clear all loaded samples."""
        if not self.samples:
            messagebox.showinfo("No Samples", "No samples to clear.")
            return
        
        response = messagebox.askyesno(
            "Confirm Clear All",
            f"Delete all {len(self.samples)} samples?\n\nThis cannot be undone."
        )
        
        if response:
            self.samples = []
            self.current_text = ""
            self.current_index = 0
            self.text_display.delete(1.0, tk.END)
            self.update_sample_list()
            self.status_label.config(text="All samples cleared", fg="orange")
    
    def save_current_text(self):
        """Save the current text to a new file."""
        if not self.current_text:
            messagebox.showwarning("No Text", "No text to save.")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Text As",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self.current_text)
                
                messagebox.showinfo("Success", f"Text saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{str(e)}")
    
    def next_sample(self):
        """Move to next sample."""
        if self.samples and self.current_index < len(self.samples) - 1:
            self.current_index += 1
            self.sample_listbox.selection_clear(0, tk.END)
            self.sample_listbox.selection_set(self.current_index)
            self.sample_listbox.see(self.current_index)
            self.on_sample_select(None)
    
    def previous_sample(self):
        """Move to previous sample."""
        if self.samples and self.current_index > 0:
            self.current_index -= 1
            self.sample_listbox.selection_clear(0, tk.END)
            self.sample_listbox.selection_set(self.current_index)
            self.sample_listbox.see(self.current_index)
            self.on_sample_select(None)
    
    def process_queue(self):
        """Process messages from background threads."""
        try:
            while True:
                msg_type, msg_data = self.message_queue.get_nowait()
                
                if msg_type == "speaking_done":
                    self.speak_btn.config(state=tk.NORMAL)
                    self.replay_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.status_label.config(text="Speech completed - Ready to replay", fg="green")
                
                elif msg_type == "test_done":
                    self.is_speaking = False
                    self.stop_btn.config(state=tk.DISABLED)
                    self.status_label.config(text="Voice test completed", fg="green")
                
                elif msg_type == "error":
                    messagebox.showerror("Error", msg_data)
                    self.speak_btn.config(state=tk.NORMAL)
                    self.replay_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.is_speaking = False
        
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_queue)
    
    def show_audio_help(self):
        """Show audio troubleshooting help."""
        help_text = """🔊 AUDIO TROUBLESHOOTING

If you can't hear the computer voice:

1. CHECK VOLUME:
   - Move volume slider to 100%
   - Increase your computer/speaker volume
   - Check Windows volume mixer

2. CHECK SPEAKERS:
   - Are speakers/headphones plugged in?
   - Are they turned on?
   - Try playing music to test

3. CHECK WINDOWS SETTINGS:
   - Right-click speaker icon in taskbar
   - Select "Open Sound settings"
   - Make sure correct output device is selected
   - Test your speakers in Windows settings

4. CHECK TTS ENGINE:
   - Click "Test Voice" button
   - You should hear: "Testing. Can you hear me?"
   - If not, restart the application

5. ALTERNATIVE:
   - The audio IS playing through your speakers
   - Use Windows "Stereo Mix" to record it
   - Or use gui_trainer.py to record while this plays

Still not working?
- Restart your computer
- Check audio drivers are installed
- Try different speakers/headphones
"""
        
        messagebox.showinfo("Audio Troubleshooting", help_text)


def main():
    """Main entry point."""
    root = tk.Tk()
    app = SampleGenerator(root)
    root.mainloop()


if __name__ == "__main__":
    main()