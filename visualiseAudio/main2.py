import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from spleeter.separator import Separator

def process_audio():
    # Load mono audio at 8kHz for only 10 seconds
    filename = 'visualiseAudio/letdown.mp3'
    y, sr = librosa.load(filename, sr=8000, mono=True, duration=10.0)

    # Try low-memory STFT
    try:
        D = librosa.stft(y, n_fft=256, hop_length=128)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram (Reduced Load)')
        plt.tight_layout()
        plt.show()
    except MemoryError:
        print("⚠️ Not enough memory to display spectrogram. Skipping visualization.")

    # Run Spleeter
    print("Running Spleeter source separation...")
    separator = Separator('spleeter:4stems')
    output_path = 'spleeter_output'
    os.makedirs(output_path, exist_ok=True)
    separator.separate_to_file(filename, output_path)
    print(f"✅ Done. Check: {os.path.join(output_path, os.path.splitext(os.path.basename(filename))[0])}")

if __name__ == '__main__':
    process_audio()
