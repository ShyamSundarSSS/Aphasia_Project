"""
Batch Audio Generator using System Audio Recording
This script plays aphasia samples through speakers which can be recorded
by the training GUI or system audio recorder.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import time
from pathlib import Path

# Import your working sample generator
from sample_generator import SampleGenerator

class BatchAudioGenerator:
    """Generates multiple aphasia samples by playing them through speakers."""
    
    def __init__(self, root):
        """Initialize the batch generator."""
        self.root = root
        self.root.title("Batch Aphasia Audio Generator")
        self.root.geometry("600x400")
        
        # Use the working sample generator
        self.generator_root = tk.Toplevel(root)
        self.generator = SampleGenerator(self.generator_root)
        self.generator_root.withdraw()  # Hide the generator window
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI."""
        title = tk.Label(
            self.root,
            text="Batch Aphasia Audio Generator",
            font=("Arial", 14, "bold"),
            bg="#4CAF50",
            fg="white",
            pady=10
        )
        title.pack(fill=tk.X)
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        info = ttk.Label(
            main_frame,
            text="This tool will play aphasia samples through your speakers.\n"
                 "IMPORTANT: Start your training GUI first and click 'Start Recording'\n"
                 "before generating each sample.",
            font=("Arial", 9),
            justify=tk.CENTER,
            foreground="red"
        )
        info.pack(pady=10)
        
        # Aphasia type
        type_frame = ttk.LabelFrame(main_frame, text="Select Aphasia Type", padding="10")
        type_frame.pack(fill=tk.X, pady=5)
        
        self.aphasia_type = tk.StringVar(value='broca')
        types = [
            ("Normal Speech", "normal"),
            ("Mild Aphasia", "mild"),
            ("Broca's (Nonfluent)", "broca"),
            ("Wernicke's (Fluent)", "wernicke"),
            ("Severe Aphasia", "severe"),
            ("Very Severe Aphasia", "very_severe")
        ]
        
        for i, (label, value) in enumerate(types):
            ttk.Radiobutton(
                type_frame,
                text=label,
                variable=self.aphasia_type,
                value=value
            ).grid(row=i//2, column=i%2, sticky=tk.W, padx=10, pady=2)
        
        # Settings
        settings_frame = ttk.Frame(main_frame)
        settings_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(settings_frame, text="Number of samples:").pack(side=tk.LEFT, padx=5)
        self.sample_count = tk.IntVar(value=3)
        ttk.Spinbox(settings_frame, from_=1, to=10, textvariable=self.sample_count, width=10).pack(side=tk.LEFT)
        
        ttk.Label(settings_frame, text="Delay between samples (seconds):").pack(side=tk.LEFT, padx=5)
        self.delay = tk.IntVar(value=5)
        ttk.Spinbox(settings_frame, from_=3, to=30, textvariable=self.delay, width=10).pack(side=tk.LEFT)
        
        # Enable simulation
        self.enable_simulation = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            main_frame,
            text="Enable Aphasia Simulation (realistic pauses and hesitations)",
            variable=self.enable_simulation
        ).pack(pady=5)
        
        # Generate button
        ttk.Button(
            main_frame,
            text="🎤 Start Generating (Play through speakers)",
            command=self.start_generation,
            width=40
        ).pack(pady=10)
        
        # Status
        self.status_label = tk.Label(
            main_frame,
            text="Ready. Make sure training GUI is recording!",
            font=("Arial", 10),
            fg="green"
        )
        self.status_label.pack(pady=5)
    
    def start_generation(self):
        """Start the batch generation process."""
        count = self.sample_count.get()
        delay = self.delay.get()
        aphasia_key = self.aphasia_type.get()
        
        # Map aphasia type key to sample library naming
        type_map = {
            'normal': 'normal_speech',
            'mild': 'mild_aphasia',
            'broca': 'broca_pattern',
            'wernicke': 'wernicke_pattern',
            'severe': 'severe_aphasia',
            'very_severe': 'very_severe_aphasia'
        }
        
        response = messagebox.askyesno(
            "Start Generation",
            f"This will play {count} {self.aphasia_type.get()} samples.\n\n"
            f"MAKE SURE:\n"
            f"1. Training GUI is open\n"
            f"2. Click 'Start Recording' in training GUI BEFORE each sample\n"
            f"3. Set correct severity level in training GUI\n\n"
            f"Delay between samples: {delay} seconds\n\n"
            f"Ready to start?"
        )
        
        if not response:
            return
        
        # Load samples if not already loaded
        if not self.generator.samples:
            self.generator.load_sample_library()
        
        # Enable simulation
        self.generator.simulate_aphasia.set(self.enable_simulation.get())
        self.generator.aphasia_type.set("Auto-detect from text")
        
        # Filter samples by type
        prefix = type_map.get(aphasia_key, aphasia_key)
        matching_samples = [s for s in self.generator.samples if prefix in s['filename'].lower()]
        
        if not matching_samples:
            messagebox.showerror("Error", f"No samples found for type: {aphasia_key}")
            return
        
        # Generate each sample
        for i in range(min(count, len(matching_samples))):
            sample = matching_samples[i]
            
            self.status_label.config(
                text=f"Sample {i+1}/{count}: {sample['filename']}\nGet ready to record...",
                fg="orange"
            )
            self.root.update()
            
            # Countdown
            for countdown in range(3, 0, -1):
                self.status_label.config(text=f"Starting in {countdown}...", fg="red")
                self.root.update()
                time.sleep(1)
            
            # Select and speak the sample
            self.generator.current_index = self.generator.samples.index(sample)
            self.generator.current_text = sample['text']
            
            self.status_label.config(
                text=f"🔊 SPEAKING NOW - Sample {i+1}/{count}\n'{sample['text'][:50]}...'",
                fg="red"
            )
            self.root.update()
            
            # Speak the sample
            self.generator.speak_current()
            
            # Wait for speech to complete
            while self.generator.is_speaking:
                time.sleep(0.1)
                self.root.update()
            
            if i < count - 1:
                self.status_label.config(
                    text=f"Sample {i+1} complete. Next sample in {delay} seconds...",
                    fg="blue"
                )
                self.root.update()
                time.sleep(delay)
        
        self.status_label.config(
            text=f"✓ All {count} samples completed!",
            fg="green"
        )
        
        messagebox.showinfo(
            "Complete",
            f"Generated {count} audio samples!\n\n"
            f"Check your training_data folder for recorded files."
        )


def main():
    """Main entry point."""
    root = tk.Tk()
    app = BatchAudioGenerator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
