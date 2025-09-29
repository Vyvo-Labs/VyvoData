#!/usr/bin/env python

from vyvodata.pipelines.aes_pipeline import AudioAestheticsPredictor
import os

# Initialize once
predictor = AudioAestheticsPredictor(checkpoint_pth="facebook/audiobox-aesthetics")

# Example 1: Single audio file
audio_file = "path/to/audio.wav"  # Replace with real path
if os.path.exists(audio_file):
    single_result = predictor(audio_file)

# Example 2: Multiple audio files
audio_files = ["path/to/audio1.wav", "path/to/audio2.wav"]
multi_results = predictor(audio_files)

# Example 3: HuggingFace dataset
hf_results = predictor(
    "OpenSpeechHub/2M-Belebele-Ja",
    output_dir="./output",
    num_samples=5,
    split="train",
    save_json=True,
)
