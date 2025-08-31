import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

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

def plot_mfcc(y, sr, output_path):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_chromagram(y, sr, output_path):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('Chromagram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_tempogram(y, sr, output_path):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(tempogram, sr=sr, x_axis='time', y_axis='tempo')
    plt.colorbar()
    plt.title('Tempogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_onset_envelope(y, sr, output_path):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    plt.plot(librosa.times_like(onset_env, sr=sr), onset_env, label='Onset strength')
    plt.xlabel('Time (s)')
    plt.ylabel('Onset strength')
    plt.title('Onset Envelope')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    audio_file = "HarryPatch.mp3"  # Replace with your audio file path

    if not os.path.isfile(audio_file):
        print(f"Audio file '{audio_file}' not found.")
    else:
        y, sr = librosa.load(audio_file, sr=None)

        plot_waveform(y, sr, "waveform.png")
        plot_spectrogram(y, sr, "spectrogram2.png")
        plot_mel_spectrogram(y, sr, "mel_spectrogram.png")
        plot_mfcc(y, sr, "mfcc.png")
        plot_chromagram(y, sr, "chromagram.png")
        plot_tempogram(y, sr, "tempogram.png")
        plot_onset_envelope(y, sr, "onset_envelope.png")

        print("All diagrams have been saved.")