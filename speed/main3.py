# audio_waveform_playground_mp3.py
# A complete beginner-friendly audio manipulation script for MP3 files

import os
from pydub import AudioSegment

import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from tempfile import mktemp

# 1. Setup - Configure FFmpeg (required for MP3)
# Make sure FFmpeg is installed: https://ffmpeg.org/
AudioSegment.converter = "ffmpeg"  # Path to ffmpeg if not in system PATH

# Input/Output configuration
input_file = "audio.mp3"  # Change this to your MP3 file
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def load_and_play_mp3(file_path):
    """Load MP3 file and return audio segment"""
    # Convert MP3 to WAV temporarily for playback
    wav_path = mktemp('.wav')
    audio = AudioSegment.from_mp3(file_path)
    audio.export(wav_path, format="wav")
    
    print(f"Loaded: {file_path} ({len(audio)/1000:.1f}s, {audio.frame_rate}Hz, {audio.channels} channels)")
    
    # Play audio

    return audio, wav_path

# 2. Basic Manipulations
def manipulate_audio(audio):
    """Apply various audio effects to MP3"""
    # Speed changes
    fast_audio = audio.speedup(playback_speed=1.5)
    slow_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * 0.75)
    })
    
    # Reverse audio
    reversed_audio = audio.reverse()
    
    # Fade effects
    faded_audio = audio.fade_in(2000).fade_out(3000)  # 2s fade in, 3s fade out
    
    # Volume changes
    louder_audio = audio + 6  # +6dB
    quieter_audio = audio - 3  # -3dB
    
    return {
        "fast": fast_audio,
        "slow": slow_audio,
        "reversed": reversed_audio,
        "faded": faded_audio,
        "louder": louder_audio,
        "quieter": quieter_audio
    }

# 3. Advanced Manipulations (using librosa)
def advanced_manipulations(file_path):
    """Advanced effects needing librosa"""
    y, sr = librosa.load(file_path, sr=None)
    
    # Pitch shift
    y_shifted_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=5)
    y_shifted_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-5)
    
    # Low-pass filter
    def apply_lowpass(y, sr, cutoff=1000):
        n = len(y)
        fft = np.fft.rfft(y)
        freq = np.fft.rfftfreq(n, d=1/sr)
        fft[freq > cutoff] = 0
        return np.fft.irfft(fft)
    
    y_lowpass = apply_lowpass(y, sr)
    
    return {
        "pitch_up": (y_shifted_up, sr),
        "pitch_down": (y_shifted_down, sr),
        "lowpass": (y_lowpass, sr)
    }

# 4. Visualization
def plot_waveform(audio, title="Waveform"):
    """Plot the audio waveform"""
    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
    
    plt.figure(figsize=(12, 4))
    plt.title(title)
    
    if audio.channels == 2:
        plt.plot(samples[:, 0], label="Left", alpha=0.7)
        plt.plot(samples[:, 1], label="Right", alpha=0.7)
        plt.legend()
    else:
        plt.plot(samples)
    
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()

# 5. Main Execution
if __name__ == "__main__":
    print("=== MP3 Audio Manipulation Playground ===")
    
    # Load and play original
    original_audio,  wav_path = load_and_play_mp3(input_file)
  
    
    # Basic manipulations
    print("\nApplying basic manipulations...")
    manipulated = manipulate_audio(original_audio)
    
    # Save basic manipulations
    for name, audio in manipulated.items():
        output_path = os.path.join(output_dir, f"{name}.mp3")
        audio.export(output_path, format="mp3", bitrate="192k")
        print(f"Saved: {output_path}")
    
    # Advanced manipulations
    print("\nApplying advanced manipulations...")
    advanced = advanced_manipulations(input_file)
    
    # Save advanced manipulations
    for name, (y, sr) in advanced.items():
        output_path = os.path.join(output_dir, f"{name}.wav")  # librosa outputs to WAV
        sf.write(output_path, y, sr)
        print(f"Saved: {output_path}")
    
    # Visualize
    print("\nOriginal waveform:")
    plot_waveform(original_audio)
    
    print("\nFast version waveform:")
    plot_waveform(manipulated["fast"], "Fast (1.5x speed)")
    
    # Clean up temporary WAV file
    try:
        os.remove(wav_path)
    except:
        pass
    
    print("\nAll operations completed! Check the 'output' folder for results.")