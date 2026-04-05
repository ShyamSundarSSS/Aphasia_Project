"""
GUI Interface for Aphasia Audio Sample Generator
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path
import threading
from aphasia_audio_generator import AphasiaAudioGenerator

class AudioGeneratorGUI:
    """GUI for generating aphasia audio samples."""
    
    def __init__(self, root):
        """Initialize GUI."""
        self.root = root
        self.root.title("Aphasia Audio Sample Generator")
        self.root.geometry("700x600")
        
        self.generator = AphasiaAudioGenerator()
        self.is_generating = False
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI layout."""
        # Title
        title = tk.Label(
            self.root,
            text="Aphasia Audio Sample Generator",
            font=("Arial", 14, "bold"),
            bg="#4CAF50",
            fg="white",
            pady=10
        )
        title.pack(fill=tk.X)
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Aphasia type selection
        type_frame = ttk.LabelFrame(main_frame, text="Aphasia Type", padding="10")
        type_frame.pack(fill=tk.X, pady=5)
        
        self.aphasia_type = tk.StringVar(value='broca')
        
        types = [
            ('Normal Speech', 'normal'),
            ('Mild Aphasia', 'mild'),
            ("Broca's (Nonfluent)", 'broca'),
            ("Wernicke's (Fluent)", 'wernicke'),
            ('Severe Aphasia', 'severe'),
            ('Very Severe Aphasia', 'very_severe')
        ]
        
        for i, (label, value) in enumerate(types):
            ttk.Radiobutton(
                type_frame,
                text=label,
                variable=self.aphasia_type,
                value=value,
                command=self.update_description
            ).grid(row=i//2, column=i%2, sticky=tk.W, padx=10, pady=2)
        
        # Description
        desc_frame = ttk.LabelFrame(main_frame, text="Description", padding="10")
        desc_frame.pack(fill=tk.X, pady=5)
        
        self.description_label = tk.Label(
            desc_frame,
            text="",
            font=("Arial", 9),
            justify=tk.LEFT,
            wraplength=650
        )
        self.description_label.pack()
        
        # Settings
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(settings_frame, text="Voice Gender:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.voice_gender = tk.StringVar(value='female')
        ttk.Radiobutton(settings_frame, text="Female", variable=self.voice_gender, value='female').grid(row=0, column=1, padx=5)
        ttk.Radiobutton(settings_frame, text="Male", variable=self.voice_gender, value='male').grid(row=0, column=2, padx=5)
        
        ttk.Label(settings_frame, text="Samples to Generate:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.sample_count = tk.IntVar(value=3)
        ttk.Spinbox(settings_frame, from_=1, to=10, textvariable=self.sample_count, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Custom text
        custom_frame = ttk.LabelFrame(main_frame, text="Custom Text (Optional)", padding="10")
        custom_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.custom_text = scrolledtext.ScrolledText(custom_frame, height=4, wrap=tk.WORD, font=("Arial", 10))
        self.custom_text.pack(fill=tk.BOTH, expand=True)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.generate_btn = ttk.Button(
            button_frame,
            text="🎤 Generate Samples",
            command=self.generate_samples,
            width=20
        )
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="📋 List Templates",
            command=self.show_templates,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="📂 Open Output Folder",
            command=self.open_output_folder,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_label = tk.Label(
            main_frame,
            text="Ready to generate audio samples",
            font=("Arial", 9),
            fg="green"
        )
        self.status_label.pack(pady=5)
        
        # Initialize description
        self.update_description()
    
    def update_description(self):
        """Update description label based on selected type."""
        aphasia_type = self.aphasia_type.get()
        config = self.generator.TEMPLATES[aphasia_type]
        
        desc_text = f"{config['name']}\n\n{config['description']}\n\nExample: \"{config['samples'][0]}\""
        self.description_label.config(text=desc_text)
    
    def generate_samples(self):
        """Generate audio samples."""
        if self.is_generating:
            messagebox.showinfo("Busy", "Already generating samples. Please wait.")
            return
        
        self.is_generating = True
        self.generate_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Generating samples...", fg="orange")
        
        thread = threading.Thread(target=self._generate_thread)
        thread.daemon = True
        thread.start()
    
    def _generate_thread(self):
        """Background thread for generation."""
        try:
            aphasia_type = self.aphasia_type.get()
            gender = self.voice_gender.get()
            count = self.sample_count.get()
            custom = self.custom_text.get(1.0, tk.END).strip()
            
            if custom:
                # Generate with custom text
                self.generator.generate_sample(aphasia_type, custom_text=custom, voice_gender=gender)
                self.status_label.config(text=f"Generated 1 custom sample!", fg="green")
            else:
                # Generate batch
                self.generator.generate_batch(
                    aphasia_types=[aphasia_type],
                    samples_per_type=count,
                    voice_gender=gender
                )
                self.status_label.config(text=f"Generated {count} samples!", fg="green")
            
            messagebox.showinfo("Success", f"Generated {count} {aphasia_type} samples!\nCheck the 'generated_aphasia_samples' folder.")
        
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg="red")
            messagebox.showerror("Error", f"Generation failed:\n{str(e)}")
        
        finally:
            self.is_generating = False
            self.generate_btn.config(state=tk.NORMAL)
    
    def show_templates(self):
        """Show available templates in a new window."""
        self.generator.list_templates()
        messagebox.showinfo("Templates", "Template information printed to console.\nCheck your terminal/command prompt.")
    
    def open_output_folder(self):
        """Open output folder in file explorer."""
        import os
        import platform
        
        folder = str(self.generator.output_dir.absolute())
        
        try:
            if platform.system() == "Windows":
                os.startfile(folder)
            elif platform.system() == "Darwin":  # macOS
                os.system(f'open "{folder}"')
            else:  # Linux
                os.system(f'xdg-open "{folder}"')
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder:\n{str(e)}")


def main():
    """Main entry point."""
    root = tk.Tk()
    app = AudioGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
