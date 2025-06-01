import librosa
import numpy as np
import soundfile as sf
import os
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def basic_vocal_isolation(audio_file):
    """
    Basic method: Center channel extraction for vocal isolation
    Works best with stereo tracks where vocals are centered
    """
    print("Loading audio file...")
    # Load stereo audio
    y, sr = librosa.load(audio_file, sr=None, mono=False)
    
    if len(y.shape) == 1:
        print("Audio is mono, converting to fake stereo for demonstration")
        y = np.stack([y, y])
    
    left, right = y[0], y[1]
    
    # Center channel extraction
    vocals_isolated = (left + right) / 2  # Center channel (vocals)
    instrumental = (left - right) / 2     # Side channels (instruments)
    
    # Save outputs
    sf.write('vocals_basic.wav', vocals_isolated, sr)
    sf.write('instrumental_basic.wav', instrumental, sr)
    
    print(f"Basic separation complete!")
    print(f"- Vocals saved to: vocals_basic.wav")
    print(f"- Instrumental saved to: instrumental_basic.wav")
    
    return vocals_isolated, instrumental, sr

def advanced_spectral_separation(audio_file):
    """
    More advanced method using spectral analysis and masking
    Separates based on frequency characteristics and harmonic content
    """
    print("\nPerforming advanced spectral separation...")
    
    # Load audio
    y, sr = librosa.load(audio_file, sr=22050)  # Standard sample rate
    
    # Compute spectrograms
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Separate vocals using harmonic-percussive separation
    harmonic, percussive = librosa.decompose.hpss(stft, margin=3.0)
    
    # Create masks for different components
    # Vocal mask: focus on harmonic content in vocal frequency range
    vocal_mask = create_vocal_mask(magnitude, sr)
    
    # Drum mask: focus on percussive content
    drum_mask = create_drum_mask(magnitude, sr)
    
    # Apply masks
    vocals_stft = harmonic * vocal_mask
    drums_stft = percussive * drum_mask
    bass_stft = harmonic * create_bass_mask(magnitude, sr)
    other_stft = stft - vocals_stft - drums_stft - bass_stft
    
    # Convert back to time domain
    vocals = librosa.istft(vocals_stft, hop_length=512)
    drums = librosa.istft(drums_stft, hop_length=512)
    bass = librosa.istft(bass_stft, hop_length=512)
    other = librosa.istft(other_stft, hop_length=512)
    
    # Save all components
    sf.write('vocals_advanced.wav', vocals, sr)
    sf.write('drums_advanced.wav', drums, sr)
    sf.write('bass_advanced.wav', bass, sr)
    sf.write('other_advanced.wav', other, sr)
    
    print(f"Advanced separation complete!")
    print(f"- Vocals: vocals_advanced.wav")
    print(f"- Drums: drums_advanced.wav") 
    print(f"- Bass: bass_advanced.wav")
    print(f"- Other: other_advanced.wav")
    
    return vocals, drums, bass, other, sr

def create_vocal_mask(magnitude, sr):
    """Create a mask that emphasizes vocal frequencies (80Hz - 8kHz)"""
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    # Vocal frequency range
    vocal_low = 80
    vocal_high = 8000
    
    mask = np.zeros_like(magnitude)
    vocal_bins = np.where((freqs >= vocal_low) & (freqs <= vocal_high))[0]
    
    # Smooth vocal mask with emphasis on 200Hz-4kHz
    for i, freq in enumerate(freqs):
        if vocal_low <= freq <= vocal_high:
            # Gaussian-like emphasis on vocal formants
            emphasis = np.exp(-((freq - 1000) / 1500) ** 2)
            mask[i, :] = emphasis
    
    return mask

def create_drum_mask(magnitude, sr):
    """Create a mask for drum frequencies"""
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    mask = np.zeros_like(magnitude)
    
    # Kick drum: 20-100Hz
    kick_bins = np.where((freqs >= 20) & (freqs <= 100))[0]
    # Snare: 150-250Hz and 2-8kHz
    snare_bins1 = np.where((freqs >= 150) & (freqs <= 250))[0]
    snare_bins2 = np.where((freqs >= 2000) & (freqs <= 8000))[0]
    # Hi-hats: 8-20kHz
    hihat_bins = np.where((freqs >= 8000) & (freqs <= 20000))[0]
    
    # Set mask values
    for bins in [kick_bins, snare_bins1, snare_bins2, hihat_bins]:
        if len(bins) > 0:
            mask[bins, :] = 0.8
    
    return mask

def create_bass_mask(magnitude, sr):
    """Create a mask for bass frequencies (20-250Hz)"""
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    mask = np.zeros_like(magnitude)
    bass_bins = np.where((freqs >= 20) & (freqs <= 250))[0]
    
    if len(bass_bins) > 0:
        mask[bass_bins, :] = 0.9
    
    return mask

def analyze_separation_quality(original_file, separated_vocals, sr):
    """Analyze the quality of separation"""
    print("\nAnalyzing separation quality...")
    
    # Load original for comparison
    original, _ = librosa.load(original_file, sr=sr)
    
    # Compute spectrograms for visualization
    orig_stft = librosa.stft(original)
    vocal_stft = librosa.stft(separated_vocals)
    
    # Calculate separation metrics
    original_rms = np.sqrt(np.mean(original**2))
    vocal_rms = np.sqrt(np.mean(separated_vocals**2))
    
    print(f"Original RMS energy: {original_rms:.4f}")
    print(f"Separated vocals RMS energy: {vocal_rms:.4f}")
    print(f"Energy ratio: {vocal_rms/original_rms:.2%}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(orig_stft)), 
                            sr=sr, x_axis='time', y_axis='hz')
    plt.title('Original Audio Spectrogram')
    plt.colorbar()
    
    plt.subplot(3, 1, 2) 
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(vocal_stft)), 
                            sr=sr, x_axis='time', y_axis='hz')
    plt.title('Separated Vocals Spectrogram')
    plt.colorbar()
    
    plt.subplot(3, 1, 3)
    plt.plot(np.linspace(0, len(original)/sr, len(original)), original, alpha=0.7, label='Original')
    plt.plot(np.linspace(0, len(separated_vocals)/sr, len(separated_vocals)), separated_vocals, alpha=0.7, label='Vocals')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('separation_analysis.png', dpi=300, bbox_inches='tight')
    print("Analysis plot saved as: separation_analysis.png")

def main():
    """Main function to run audio separation"""
    
    # Input file - PUT YOUR ACTUAL MP3 FILENAME HERE
    audio_file = "AudioSeperation\Radiohead - Let Down.mp3"  # <<< CHANGE THIS TO YOUR ACTUAL FILE NAME
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: File '{audio_file}' not found!")
        print("Available files in current directory:")
        for file in os.listdir('.'):
            if file.endswith(('.mp3', '.wav', '.flac', '.m4a')):
                print(f"  - {file}")
        print("\nPlease update the 'audio_file' variable with your actual filename.")
        return  # EXIT instead of creating demo
    
    print(f"Processing: {audio_file}")
    print("="*50)
    
    # Method 1: Basic vocal isolation
    try:
        vocals_basic, instrumental_basic, sr = basic_vocal_isolation(audio_file)
        print("✓ Basic separation completed")
    except Exception as e:
        print(f"✗ Basic separation failed: {e}")
        return
    
    # Method 2: Advanced spectral separation  
    try:
        vocals_adv, drums_adv, bass_adv, other_adv, sr = advanced_spectral_separation(audio_file)
        print("✓ Advanced separation completed")
    except Exception as e:
        print(f"✗ Advanced separation failed: {e}")
        return
    
    # Analysis
    try:
        analyze_separation_quality(audio_file, vocals_adv, sr)
        print("✓ Quality analysis completed")
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
    
    print("\n" + "="*50)
    print("SEPARATION COMPLETE!")
    print("\nOutput files created:")
    print("Basic method:")
    print("  - vocals_basic.wav")  
    print("  - instrumental_basic.wav")
    print("\nAdvanced method:")
    print("  - vocals_advanced.wav")
    print("  - drums_advanced.wav")
    print("  - bass_advanced.wav") 
    print("  - other_advanced.wav")
    print("\nAnalysis:")
    print("  - separation_analysis.png")

# REMOVED THE DEMO AUDIO FUNCTION - NO MORE FAKE AUDIO!

if __name__ == "__main__":
    main()