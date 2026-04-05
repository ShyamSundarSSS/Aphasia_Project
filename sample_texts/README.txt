APHASIA SAMPLE TEXT LIBRARY
============================

This directory contains text samples representing different aphasia severity levels
and types. These samples can be played through the Sample Generator tool to create
training data for the aphasia detection system.

SAMPLE NAMING CONVENTION:
- normal_speech_*.txt - Healthy control samples
- mild_aphasia_*.txt - Mild aphasia patterns
- moderate_aphasia_*.txt - Moderate aphasia patterns
- severe_aphasia_*.txt - Severe aphasia patterns
- very_severe_aphasia_*.txt - Very severe aphasia patterns
- broca_pattern_*.txt - Broca's aphasia specific patterns
- wernicke_pattern_*.txt - Wernicke's aphasia specific patterns

HOW TO USE:
1. Run: python sample_generator.py
2. Click "Load Sample Library"
3. Select a sample and click "Speak Current"
4. Record in the training GUI simultaneously

CREATING NEW SAMPLES:
Simply create new .txt files in this directory following the naming convention.
The text should represent realistic aphasia speech patterns.

Example:
--------
File: moderate_aphasia_new.txt
Content: "I want... go store. Buy milk... and... bread. Tomorrow maybe."
