# Speed up the mp3 file and visualize the results
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import soundfile as sf

# def change_speed(input_file, output_file, speed_factor):
#     # Load the audio file
#     y, sr = librosa.load(input_file, sr=None)
    
#     # Change the speed (time-stretch)
#     y_fast = librosa.effects.time_stretch(y, rate=speed_factor)
    
#     # Save the modified audio to a new file
#     sf.write(output_file, y_fast, sr)
    
#     return y_fast, sr

def naive_time_stretch(y, sr, speed_factor):
    # 1. Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)
    # 2. Phase vocoder time-stretch
    D_stretched = librosa.phase_vocoder(D, rate=speed_factor)
    # 3. Inverse STFT to get back to time domain
    y_stretched = librosa.istft(D_stretched, length=int(len(y) / speed_factor))
    return y_stretched

def change_speed(input_file, output_file, speed_factor):
    y, sr = librosa.load(input_file, sr=None)
    y_fast = naive_time_stretch(y, sr, speed_factor)
    sf.write(output_file, y_fast, sr)
    return y_fast, sr

def play_audio(file_path):
    sound = AudioSegment.from_file(file_path)
    play(sound)

def plot_waveform(y, sr, output_path):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_spectrogram(y, sr, output_path):
    D = np.abs(librosa.stft(y))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_mel_spectrogram(y, sr, output_path):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    input_file = "HarryPatch.mp3"  # Replace with your input file
    output_file = "HarryPatch_fast.wav"
    speed_factor = 1.5  # >1 speeds up, <1 slows down

    if not os.path.isfile(input_file):
        print(f"Audio file '{input_file}' not found.")
    else:
        # Change speed and save
        y_fast, sr = change_speed(input_file, output_file, speed_factor)
        print(f"Saved sped-up audio to {output_file}")

        # Plot diagrams for the sped-up audio
        plot_waveform(y_fast, sr, "waveform_fast.png")
        plot_spectrogram(y_fast, sr, "spectrogram_fast.png")
        plot_mel_spectrogram(y_fast, sr, "mel_spectrogram_fast.png")
        print("Diagrams for sped-up audio have been saved.")

        # Optionally play the sped-up audio
        # play_audio(output_file)