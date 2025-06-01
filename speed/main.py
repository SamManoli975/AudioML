import librosa
import soundfile as sf
import numpy as np
from scipy.signal import fftconvolve

# Load your audio file
y, sr = librosa.load("audio.mp3", sr=None)

# --- SPEED UP / SLOW DOWN ---
speed_factor = float(input("enter speed: ")) # >1 speeds up, <1 slows down
y_fast = librosa.effects.time_stretch(y, rate=speed_factor)


# --- ADD REVERB ---
def add_reverb(audio, sr, decay=0.5):
    # Simple impulse response for reverb
    impulse = np.zeros(int(sr * 0.3))
    impulse[0] = 1.0
    impulse[int(sr * 0.05)] = decay
    impulse[int(sr * 0.1)] = decay ** 2
    impulse[int(sr * 0.15)] = decay ** 3
    return fftconvolve(audio, impulse, mode='full')[:len(audio)]

y_reverb = add_reverb(y_fast, sr)

# --- SAVE TO FILE ---
sf.write("output.wav", y_reverb, sr)
