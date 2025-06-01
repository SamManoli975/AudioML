# EXTREME AUDIO ANALYSIS - Everything you can do with one audio file!

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.spatial.distance import euclidean
import soundfile as sf

# Load your audio file
audio_file = "FirstProject\Radiohead - Let Down.mp3"  # <-- CHANGE THIS
y, sr = librosa.load(audio_file, duration=30)  # Load 30 seconds

print("🎵 EXTREME AUDIO ANALYSIS STARTING...")
print("=" * 60)

# =============================================================================
# 1. SPECTRAL ANALYSIS - SEE EVERY FREQUENCY
# =============================================================================

def spectral_analysis():
    print("\n🔬 SPECTRAL ANALYSIS - See Every Frequency")
    
    # Create spectrogram - time vs frequency vs intensity
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Spectrogram (the holy grail of audio analysis)
    plt.subplot(2, 2, 1)
    librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram - Every Frequency Over Time')
    
    # Plot 2: Mel-spectrogram (how humans hear)
    plt.subplot(2, 2, 2)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(mel_spec_db, y_axis='mel', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram - How Humans Hear')
    
    # Plot 3: Chromagram (musical notes)
    plt.subplot(2, 2, 3)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('Chromagram - Musical Notes Over Time')
    
    # Plot 4: Spectral centroid (brightness over time)
    plt.subplot(2, 2, 4)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    times = librosa.frames_to_time(np.arange(len(spec_centroid)), sr=sr)
    plt.plot(times, spec_centroid)
    plt.title('Spectral Centroid - "Brightness" Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Hz')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 2. SOURCE SEPARATION - SEPARATE INSTRUMENTS/VOCALS
# =============================================================================

def source_separation():
    print("\n🎼 SOURCE SEPARATION - Isolate Different Sounds")
    
    # Harmonic-Percussive separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Save separated tracks
    sf.write('harmonic_only.wav', y_harmonic, sr)
    sf.write('percussive_only.wav', y_percussive, sr)
    
    print("✅ Created: harmonic_only.wav (melody/vocals)")
    print("✅ Created: percussive_only.wav (drums/percussion)")
    
    # Vocal isolation (basic method)
    # This works by assuming vocals are centered (mono)
    S_full, phase = librosa.magphase(librosa.stft(y))
    S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine')
    S_filter = np.minimum(S_filter, S_full)
    margin_i, margin_v = 2, 10
    power = 2
    mask_i = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power=power)
    mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)
    
    # Apply masks
    S_foreground = mask_v * S_full
    S_background = mask_i * S_full
    
    # Convert back to audio
    y_vocals = librosa.istft(S_foreground * phase)
    y_background = librosa.istft(S_background * phase)
    
    sf.write('vocals_isolated.wav', y_vocals, sr)
    sf.write('background_music.wav', y_background, sr)
    
    print("✅ Created: vocals_isolated.wav")
    print("✅ Created: background_music.wav")
    
    # Plot separation results
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(y[:sr*5])  # First 5 seconds
    plt.title('Original Audio')
    
    plt.subplot(2, 2, 2)
    plt.plot(y_harmonic[:sr*5])
    plt.title('Harmonic Component (Melody)')
    
    plt.subplot(2, 2, 3)
    plt.plot(y_percussive[:sr*5])
    plt.title('Percussive Component (Drums)')
    
    plt.subplot(2, 2, 4)
    plt.plot(y_vocals[:sr*5])
    plt.title('Isolated Vocals')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 3. TEMPO AND BEAT TRACKING
# =============================================================================

def tempo_analysis():
    print("\n🥁 TEMPO & BEAT ANALYSIS")
    
    # Extract tempo and beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    print(f"🎵 Detected Tempo: {tempo:.1f} BPM")
    print(f"🎵 Number of beats: {len(beats)}")
    
    # Onset detection (when new sounds start)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    print(f"🎵 Number of onsets: {len(onset_frames)}")
    
    # Plot tempo analysis
    plt.figure(figsize=(15, 6))
    
    plt.subplot(2, 1, 1)
    times = librosa.frames_to_time(np.arange(len(y)), sr=sr)
    plt.plot(times, y, alpha=0.6, label='Audio')
    plt.vlines(beat_times, -1, 1, color='red', alpha=0.8, label='Beats')
    plt.vlines(onset_times, -1, 1, color='green', alpha=0.6, label='Onsets')
    plt.title(f'Beat Tracking - Tempo: {tempo:.1f} BPM')
    plt.legend()
    plt.xlim(0, 10)  # First 10 seconds
    
    # Tempogram (tempo over time)
    plt.subplot(2, 1, 2)
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo')
    plt.colorbar()
    plt.title('Tempogram - Tempo Over Time')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 4. PITCH AND HARMONY ANALYSIS
# =============================================================================

def pitch_analysis():
    print("\n🎼 PITCH & HARMONY ANALYSIS")
    
    # Extract fundamental frequency (pitch)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    # Get the most prominent pitch at each time frame
    pitch_track = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        pitch_track.append(pitch)
    
    pitch_track = np.array(pitch_track)
    
    # Key detection
    y_harmonic = librosa.effects.hpss(y)[0]
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    
    # Estimate key using chroma
    chroma_mean = np.mean(chroma, axis=1)
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    estimated_key = key_names[np.argmax(chroma_mean)]
    
    print(f"🎼 Estimated Key: {estimated_key}")
    print(f"🎼 Average Pitch: {np.mean(pitch_track[pitch_track > 0]):.1f} Hz")
    
    # Plot pitch analysis
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    times = librosa.frames_to_time(np.arange(len(pitch_track)), sr=sr)
    plt.plot(times, pitch_track)
    plt.title('Pitch Track - Fundamental Frequency Over Time')
    plt.ylabel('Frequency (Hz)')
    plt.xlim(0, 10)
    
    plt.subplot(2, 1, 2)
    plt.bar(key_names, chroma_mean)
    plt.title(f'Key Profile - Estimated Key: {estimated_key}')
    plt.ylabel('Chroma Intensity')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 5. ADVANCED SPECTRAL FEATURES
# =============================================================================

def advanced_features():
    print("\n🔍 ADVANCED SPECTRAL FEATURES")
    
    # Extract tons of features
    features = {}
    
    # MFCCs (most important for ML)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfccs'] = mfccs
    
    # Spectral features
    features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y)
    features['rms_energy'] = librosa.feature.rms(y=y)
    
    # Tonal features
    features['chroma'] = librosa.feature.chroma_stft(y=y, sr=sr)
    features['tonnetz'] = librosa.feature.tonnetz(y=librosa.effects.hpss(y)[0], sr=sr)
    
    # Contrast and novelty
    S = np.abs(librosa.stft(y))
    features['spectral_contrast'] = librosa.feature.spectral_contrast(S=S, sr=sr)
    
    print("✅ Extracted features:")
    for name, feature in features.items():
        print(f"   {name}: {feature.shape}")
    
    # Plot feature summary
    plt.figure(figsize=(20, 12))
    
    # Plot first 12 MFCCs
    for i in range(12):
        plt.subplot(4, 3, i+1)
        librosa.display.specshow(mfccs[i:i+1], x_axis='time', sr=sr)
        plt.title(f'MFCC {i+1}')
        plt.colorbar()
    
    plt.tight_layout()
    plt.show()

    return features

# =============================================================================
# 6. AUDIO EFFECTS AND TRANSFORMATIONS
# =============================================================================

def audio_effects():
    print("\n🎛️  AUDIO EFFECTS & TRANSFORMATIONS")
    
    # Time stretching (change speed without changing pitch)
    y_fast = librosa.effects.time_stretch(y, rate=1.5)
    y_slow = librosa.effects.time_stretch(y, rate=0.7)
    
    sf.write('audio_fast.wav', y_fast, sr)
    sf.write('audio_slow.wav', y_slow, sr)
    print("✅ Created: audio_fast.wav (1.5x speed)")
    print("✅ Created: audio_slow.wav (0.7x speed)")
    
    # Pitch shifting (change pitch without changing speed)
    y_higher = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)  # 4 semitones up
    y_lower = librosa.effects.pitch_shift(y, sr=sr, n_steps=-4)  # 4 semitones down
    
    sf.write('audio_higher.wav', y_higher, sr)
    sf.write('audio_lower.wav', y_lower, sr)
    print("✅ Created: audio_higher.wav (+4 semitones)")
    print("✅ Created: audio_lower.wav (-4 semitones)")
    
    # Add reverb effect
    y_reverb = np.copy(y)
    for delay in [0.1, 0.2, 0.3]:  # Multiple delays
        delay_samples = int(delay * sr)
        if delay_samples < len(y):
            y_reverb[delay_samples:] += 0.3 * y[:-delay_samples]
    
    sf.write('audio_reverb.wav', y_reverb, sr)
    print("✅ Created: audio_reverb.wav (with reverb)")

# =============================================================================
# 7. SIMILARITY AND COMPARISON
# =============================================================================

def audio_similarity():
    print("\n🔍 AUDIO SIMILARITY ANALYSIS")
    
    # Split audio into segments
    segment_length = 5 * sr  # 5 second segments
    segments = []
    
    for i in range(0, len(y) - segment_length, segment_length):
        segment = y[i:i + segment_length]
        # Extract features for each segment
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        segments.append(mfcc_mean)
    
    # Calculate similarity matrix
    n_segments = len(segments)
    similarity_matrix = np.zeros((n_segments, n_segments))
    
    for i in range(n_segments):
        for j in range(n_segments):
            # Use euclidean distance (lower = more similar)
            similarity_matrix[i, j] = euclidean(segments[i], segments[j])
    
    # Plot similarity matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar(label='Distance (lower = more similar)')
    plt.title('Self-Similarity Matrix - Which Parts Sound Similar?')
    plt.xlabel('Segment Number')
    plt.ylabel('Segment Number')
    plt.show()
    
    print(f"✅ Analyzed {n_segments} segments of 5 seconds each")
    print("Dark squares = similar segments, bright squares = different segments")

# =============================================================================
# RUN ALL ANALYSES
# =============================================================================

if __name__ == "__main__":
    
    print("Choose what to run (or run all):")
    print("1. Spectral Analysis")
    print("2. Source Separation (isolate vocals/instruments)")
    print("3. Tempo & Beat Analysis")
    print("4. Pitch & Harmony Analysis")
    print("5. Advanced Features Extraction")
    print("6. Audio Effects & Transformations")
    print("7. Similarity Analysis")
    print("8. RUN EVERYTHING!")
    
    choice = input("\nEnter choice (1-8): ")
    
    if choice == '1' or choice == '8':
        spectral_analysis()
    
    if choice == '2' or choice == '8':
        source_separation()
    
    if choice == '3' or choice == '8':
        tempo_analysis()
    
    if choice == '4' or choice == '8':
        pitch_analysis()
    
    if choice == '5' or choice == '8':
        features = advanced_features()
    
    if choice == '6' or choice == '8':
        audio_effects()
    
    if choice == '7' or choice == '8':
        audio_similarity()
    
    print("\n" + "=" * 60)
    print("🎉 ANALYSIS COMPLETE!")
    print("Check your folder for generated audio files!")
    print("🎵 You just did professional-level audio analysis!")