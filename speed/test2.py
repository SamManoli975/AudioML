import numpy as np
import matplotlib.pyplot as plt

# Create a 440Hz sine wave (1 second, 44100Hz sample rate)
sr = 44100  # Sample rate
t = np.linspace(0, 1, sr, endpoint=False)  # Time array
frequency = 440  # Hz
amplitude = 0.5  # -6dB
sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)


speed_factor = 1.5
# Time-domain resampling (crude method for illustration)
sped_up = np.interp(
    np.arange(0, len(sine_wave), speed_factor),
    np.arange(len(sine_wave)),
    sine_wave
)

# plt.figure(figsize=(10,4))
# plt.plot(t[:500], sine_wave[:500], label="Original (440Hz)")
# plt.plot(t[:333], sped_up[:333], label="1.5x Speed (~660Hz)")  # 500/1.5 ≈ 333
# plt.legend()
# plt.title("Speed Change: Frequency Increases")
# plt.show()

fade_duration = 0.2  # seconds
fade_samples = int(sr * fade_duration)
fade_in = np.ones_like(sine_wave)
fade_in[:fade_samples] = np.linspace(0, 1, fade_samples)  # Linear fade
faded_wave = sine_wave * fade_in

# plt.plot(t[:1000], faded_wave[:1000])
# plt.title("Fade In (First 200ms)")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.show()

n_steps = 5  # Semitones to shift
pitch_factor = 2 ** (n_steps / 12)  # 12 semitones = 1 octave
# Librosa's pitch shift (using phase vocoder)
import librosa
y_shifted = librosa.effects.pitch_shift(sine_wave, sr=sr, n_steps=5)

plt.plot(t[:500], sine_wave[:500], label="Original (440Hz)")
plt.plot(t[:500], y_shifted[:500], label="Pitch Up (~659Hz)", alpha=0.7)  # 440 * 2^(5/12) ≈ 659Hz
plt.legend()
plt.title("Pitch Shift: More Wave Cycles in Same Time")
plt.show()

db_change = 6
louder_wave = sine_wave * (10 ** (db_change / 20))  # ≈ 2.0x multiplier

plt.plot(t[:500], sine_wave[:500], label="Original (0.5 amplitude)")
plt.plot(t[:500], louder_wave[:500], label="+6dB (1.0 amplitude)", alpha=0.7)
plt.legend()
plt.title("Loudness Change: Amplitude Scaling")
plt.show()