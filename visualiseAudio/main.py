import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from scipy.signal.windows import hamming
import scipy.fftpack
import soundfile as sf

# Load the audio
filename = "visualiseAudio\Radiohead - Let Down.mp3"
y, sr = librosa.load(filename, sr=None)

# Extract audio features
duration = librosa.get_duration(y=y, sr=sr)
times = np.linspace(0, duration, len(y))

# ZCR & RMS
zcr = librosa.feature.zero_crossing_rate(y)[0]
rms = librosa.feature.rms(y=y)[0]

# STFT
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Mel Spectrogram
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
mel_db = librosa.power_to_db(mel_spec, ref=np.max)

# MFCC
mfcc = librosa.feature.mfcc(S=mel_db, sr=sr, n_mfcc=13)

# Chroma
chroma = librosa.feature.chroma_stft(S=np.abs(D), sr=sr)

# Pitch
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
pitch = [p[np.argmax(m)] if np.max(m) > 0.1 else 0 for p, m in zip(pitches.T, magnitudes.T)]
pitch = np.array(pitch)

# Formant estimation using LPC (simplified)
def get_formants(signal, sr):
    N = len(signal)
    window = hamming(N)
    signal = signal * window
    A = librosa.lpc(signal, order=16)
    roots = np.roots(A)
    roots = [r for r in roots if np.imag(r) >= 0]
    angles = np.angle(roots)
    formants = sorted(angles * (sr / (2 * np.pi)))
    return formants[:3]

formants = get_formants(y[:2048], sr)  # just the first window

# Plotting
plt.figure(figsize=(20, 25))

# 1. Waveform
plt.subplot(5, 2, 1)
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform (Time vs Amplitude)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# 2. Spectrogram
plt.subplot(5, 2, 2)
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='linear', cmap='magma')
plt.title("Spectrogram (Time vs Frequency)")
plt.colorbar(format='%+2.0f dB')

# 3. Mel Spectrogram
plt.subplot(5, 2, 3)
librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr, cmap='coolwarm')
plt.title("Mel Spectrogram (Perceived Frequency - Mel Scale)")
plt.colorbar(format='%+2.0f dB')

# 4. Pitch Contour
plt.subplot(5, 2, 4)
plt.plot(pitch)
plt.title("Pitch Contour")
plt.xlabel("Frames")
plt.ylabel("Pitch (Hz)")

# 5. Formants
plt.subplot(5, 2, 5)
plt.bar(['F1', 'F2', 'F3'], formants)
plt.title("Formants (Resonant Frequencies)")

# 6. Zero Crossing Rate
plt.subplot(5, 2, 6)
plt.plot(zcr)
plt.title("Zero Crossing Rate (Voiced/Unvoiced Detection)")
plt.xlabel("Frames")

# 7. RMS Energy
plt.subplot(5, 2, 7)
plt.plot(rms)
plt.title("RMS Energy (Loudness Dynamics)")
plt.xlabel("Frames")

# 8. MFCC
plt.subplot(5, 2, 8)
librosa.display.specshow(mfcc, x_axis='time', sr=sr)
plt.title("MFCC (Timbre & Envelope)")
plt.colorbar()

# 9. Chroma
plt.subplot(5, 2, 9)
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', cmap='cividis', sr=sr)
plt.title("Chroma (Tonal Components)")
plt.colorbar()

# 10. Raw waveform zoom-in for visualizing small segment detail
plt.subplot(5, 2, 10)
zoom_y = y[sr*10:sr*11]
zoom_t = times[sr*10:sr*11]
plt.plot(zoom_t, zoom_y)
plt.title("Waveform Zoom (Closer look at signal detail)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
