import argparse
from pathlib import Path
from src.pipeline import AphasiaPipeline
from src.training_manager import SEVERITY_LABELS, save_sample_from_results, train_and_generate_plots
from src.demo_metrics import display_demo_metrics


def prompt_yes_no(message: str, default: bool = True) -> bool:
    prompt = "[Y/n]" if default else "[y/N]"
    while True:
        response = input(f"{message} {prompt} ").strip().lower()
        if not response:
            return default
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print("Please enter 'y' or 'n'.")


def prompt_severity_label(default_label: str) -> str:
    print("\nSelect the TRUE severity level for training:")
    for i, label in enumerate(SEVERITY_LABELS, 1):
        print(f"  {i}. {label}")
    print(f"Press Enter to use predicted: {default_label}")

    while True:
        response = input("Enter choice (1-5): ").strip()
        if not response:
            return default_label
        if response.isdigit():
            idx = int(response)
            if 1 <= idx <= len(SEVERITY_LABELS):
                return SEVERITY_LABELS[idx - 1]
        print("Invalid selection. Please enter 1-5 or press Enter.")


def prompt_aphasia_type() -> str:
    response = input("Aphasia type (optional, e.g., Broca's/Wernicke's/None): ").strip()
    return response if response else "None"

def main():
    """Main entry point for the aphasia detection system."""
    parser = argparse.ArgumentParser(
        description="Aphasia Severity Assessment System - Record speech and classify severity level"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["record", "file"],
        default="record",
        help="Mode: 'record' to record from microphone, 'file' to process existing audio"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Recording duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        help="Path to audio file (required for 'file' mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="temp_audio.wav",
        help="Path to save recorded audio (default: temp_audio.wav)"
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the audio file after processing"
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--no-scenario",
        action="store_true",
        help="Skip scenario prompt in record mode"
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List all available speaking scenarios"
    )
    parser.add_argument(
        "--no-training-save",
        action="store_true",
        help="Do not save recordings to training data or generate training plots"
    )
    
    args = parser.parse_args()
    
    # List scenarios and exit if requested
    if args.list_scenarios:
        from src.scenario_generator import ScenarioGenerator
        generator = ScenarioGenerator()
        print("\n" + "="*60)
        print("AVAILABLE SPEAKING SCENARIOS")
        print("="*60)
        for i, title in enumerate(generator.list_all_scenarios(), 1):
            print(f"{i}. {title}")
        print("="*60 + "\n")
        return
    
    # Validate arguments
    if args.mode == "file" and not args.audio_file:
        parser.error("--audio-file is required when mode is 'file'")
    
    if args.mode == "file" and not Path(args.audio_file).exists():
        parser.error(f"Audio file not found: {args.audio_file}")
    
    try:
        # Initialize pipeline
        pipeline = AphasiaPipeline(whisper_model=args.whisper_model)
        
        # Process based on mode
        if args.mode == "record":
            print(f"\n{'='*60}")
            print("APHASIA SEVERITY ASSESSMENT SYSTEM")
            print(f"{'='*60}")
            print(f"Duration: {args.duration} seconds")
            print(f"Output file: {args.output}\n")
            
            # Get scenario unless disabled
            scenario = None
            if not args.no_scenario:
                scenario = pipeline.scenario_generator.get_random_scenario()
            
            results = pipeline.record_and_predict(
                duration=args.duration,
                save_path=args.output,
                scenario=scenario
            )
            
            # Display results
            pipeline.display_results(results)
            
            # Display demo evaluation metrics
            display_demo_metrics()

            if not args.no_training_save:
                if prompt_yes_no("Save this recording for training?", default=True):
                    severity_label = prompt_severity_label(results["severity_label"])
                    aphasia_type = prompt_aphasia_type()
                    save_sample_from_results(
                        results,
                        audio_source_path=args.output,
                        severity_label=severity_label,
                        aphasia_type=aphasia_type
                    )
                    trained, message = train_and_generate_plots()
                    print(message)
            
            # Cleanup
            if not args.keep_audio:
                pipeline.cleanup(args.output)
        
        else:  # file mode
            print(f"\n{'='*60}")
            print("APHASIA SEVERITY ASSESSMENT SYSTEM - FILE MODE")
            print(f"{'='*60}")
            print(f"Processing file: {args.audio_file}\n")
            
            results = pipeline.process_audio_file(args.audio_file)
            
            # Display results
            pipeline.display_results(results)
            
            # Display demo evaluation metrics
            display_demo_metrics()

            if not args.no_training_save:
                if prompt_yes_no("Save this recording for training?", default=True):
                    severity_label = prompt_severity_label(results["severity_label"])
                    aphasia_type = prompt_aphasia_type()
                    save_sample_from_results(
                        results,
                        audio_source_path=args.audio_file,
                        severity_label=severity_label,
                        aphasia_type=aphasia_type
                    )
                    trained, message = train_and_generate_plots()
                    print(message)
        
        print("\n✅ Assessment completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
