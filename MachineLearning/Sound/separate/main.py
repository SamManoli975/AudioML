#!/usr/bin/env python3
"""
Audio Source Separation Tool
Separates audio into vocals, drums, bass, and other instruments using Demucs or basic method.
"""

import os
import sys
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Only Demucs support
DEMUCS_AVAILABLE = False

try:
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    DEMUCS_AVAILABLE = True
    print("✓ Demucs available")
except ImportError:
    print("✗ Demucs not available")

class AudioSeparator:
    """Main class for audio source separation"""

    def __init__(self, method='demucs'):
        self.method = method
        self.demucs_model = None

        if method == 'demucs' and DEMUCS_AVAILABLE:
            self.demucs_model = get_model('htdemucs')
            print("Initialized Demucs with htdemucs model")
        else:
            print(f"Method {method} not available, falling back to basic separation")

    def separate_with_demucs(self, audio_file, output_dir):
        """Separate audio using Demucs"""
        if not DEMUCS_AVAILABLE or self.demucs_model is None:
            raise ValueError("Demucs not available")

        print(f"Processing {audio_file} with Demucs...")

        # Load audio
        audio, sr = librosa.load(audio_file, sr=44100, mono=False)

        # Ensure correct shape for Demucs (channels, samples)
        if audio.ndim == 1:
            audio = audio[None, :]  # Add channel dimension
        elif audio.shape[0] > audio.shape[1]:
            audio = audio.T  # Transpose if needed

        # Convert to tensor
        audio_tensor = torch.tensor(audio).float().unsqueeze(0)  # Add batch dimension

        # Separate using apply_model
        with torch.no_grad():
            separated = apply_model(self.demucs_model, audio_tensor)

        # Get stems (drums, bass, other, vocals)
        stems = ['drums', 'bass', 'other', 'vocals']

        # Save separated tracks
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, stem_name in enumerate(stems):
            stem_audio = separated[0, i].numpy()  # Remove batch dimension
            
            # Transpose back to (samples, channels) for soundfile
            if stem_audio.shape[0] < stem_audio.shape[1]:
                stem_audio = stem_audio.T

            output_file = output_dir / f"{audio_file} {stem_name}.wav"
            sf.write(output_file, stem_audio, sr)
            print(f"Saved: {output_file}")

    def basic_separation(self, audio_file, output_dir):
        """Basic separation using spectral techniques (fallback method)"""
        print(f"Using basic separation for {audio_file}")

        # Load audio
        y, sr = librosa.load(audio_file, sr=None)

        # Convert to stereo if mono
        if y.ndim == 1:
            y_stereo = np.stack([y, y])
        else:
            y_stereo = y

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Simple vocal extraction using center channel extraction
        vocals = y_stereo[0] - y_stereo[1] if y_stereo.shape[0] == 2 else y_stereo[0]
        instrumental = y_stereo[0] + y_stereo[1] if y_stereo.shape[0] == 2 else y_stereo[0]

        # Save basic separation
        sf.write(output_dir / "vocals_basic.wav", vocals, sr)
        sf.write(output_dir / "instrumental_basic.wav", instrumental, sr)

        print(f"Saved basic separation to {output_dir}")
        print("Note: For better results, install Demucs")

    def separate(self, audio_file, output_dir):
        """Main separation method"""
        if self.method == 'demucs' and DEMUCS_AVAILABLE:
            self.separate_with_demucs(audio_file, output_dir)
        else:
            self.basic_separation(audio_file, output_dir)

def main():
    input_file = "monk.mp3"
    output_dir = "separated"
    method = "demucs" if DEMUCS_AVAILABLE else "basic"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        return

    # Create separator and process
    separator = AudioSeparator(method=method)

    try:
        separator.separate(input_file, output_dir)
        print(f"\n✓ Separation complete! Check '{output_dir}' directory")
    except Exception as e:
        print(f"Error during separation: {e}")

if __name__ == "__main__":
    main()