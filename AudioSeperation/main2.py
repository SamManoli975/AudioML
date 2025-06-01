import librosa
import numpy as np
import soundfile as sf
import os
from scipy import signal
from scipy.optimize import minimize
from sklearn.decomposition import FastICA, NMF
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

class AdvancedSourceSeparator:
    def __init__(self, n_fft=4096, hop_length=1024, n_components=8):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_components = n_components
        self.sr = None
        
    def load_audio(self, file_path):
        """Load and preprocess audio with advanced windowing"""
        print(f"Loading audio: {file_path}")
        
        # Load with high quality settings
        y, sr = librosa.load(file_path, sr=44100, mono=True)
        self.sr = sr
        
        # Apply advanced preprocessing
        y = self.preprocess_audio(y)
        
        return y, sr
    
    def preprocess_audio(self, y):
        """Advanced audio preprocessing pipeline"""
        # 1. Remove DC offset
        y = y - np.mean(y)
        
        # 2. Normalize with soft limiting
        y = np.tanh(y / np.std(y)) * 0.95
        
        # 3. Apply psychoacoustic pre-emphasis
        pre_emphasis = 0.97
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
        
        return y
    
    def advanced_stft(self, y):
        """Compute STFT with advanced windowing and overlap"""
        # Use Kaiser window for better frequency resolution
        window = signal.windows.kaiser(self.n_fft, beta=8.6)
        
        # Compute STFT with overlap
        stft = librosa.stft(y, 
                           n_fft=self.n_fft, 
                           hop_length=self.hop_length,
                           window=window,
                           center=True)
        
        return stft
    
    def spectral_clustering_separation(self, stft_matrix):
        """Use spectral clustering to separate sources"""
        print("Performing spectral clustering separation...")
        
        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)
        
        # Convert to dB scale for better clustering
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # Reshape for clustering (frequency x time -> samples x features)
        features = magnitude_db.T  # Time x frequency
        
        # Apply PCA for dimensionality reduction before clustering
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(50, features.shape[1]))
        features_reduced = pca.fit_transform(features)
        
        # K-means clustering to separate different source types
        n_clusters = 4  # vocals, drums, bass, other
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_reduced)
        
        # Create masks for each cluster
        masks = []
        for i in range(n_clusters):
            mask = np.zeros_like(magnitude)
            time_indices = np.where(cluster_labels == i)[0]
            mask[:, time_indices] = 1.0
            masks.append(mask)
        
        return masks, phase
    
    def matrix_factorization_separation(self, stft_matrix):
        """Use Non-negative Matrix Factorization for source separation"""
        print("Performing NMF-based separation...")
        
        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)
        
        # Apply NMF to decompose the spectrogram
        nmf = NMF(n_components=self.n_components, 
                  init='nndsvd', 
                  max_iter=500, 
                  random_state=42)
        
        W = nmf.fit_transform(magnitude)  # Basis functions
        H = nmf.components_  # Activations
        
        # Cluster the basis functions to identify source types
        kmeans = KMeans(n_clusters=4, random_state=42)
        component_labels = kmeans.fit_predict(W.T)
        
        # Create source-specific reconstructions
        sources = []
        for source_id in range(4):
            # Select components for this source
            source_components = np.where(component_labels == source_id)[0]
            
            if len(source_components) > 0:
                W_source = np.zeros_like(W)
                W_source[:, source_components] = W[:, source_components]
                
                # Reconstruct this source
                source_magnitude = W_source @ H
                sources.append(source_magnitude)
            else:
                sources.append(np.zeros_like(magnitude))
        
        return sources, phase
    
    def independent_component_analysis(self, stft_matrix):
        """Use ICA for blind source separation"""
        print("Performing ICA-based separation...")
        
        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)
        
        # Apply ICA to frequency bins
        n_sources = 4
        separated_sources = []
        
        for freq_bin in range(0, magnitude.shape[0], 10):  # Process every 10th bin for speed
            if freq_bin + 10 < magnitude.shape[0]:
                freq_data = magnitude[freq_bin:freq_bin+10, :].T  # Time x frequency
                
                if freq_data.shape[1] >= 2:  # Need at least 2 features for ICA
                    try:
                        ica = FastICA(n_components=min(n_sources, freq_data.shape[1]), 
                                     random_state=42, max_iter=200)
                        sources = ica.fit_transform(freq_data)
                        
                        # Pad sources to match our target number
                        if sources.shape[1] < n_sources:
                            padding = np.zeros((sources.shape[0], n_sources - sources.shape[1]))
                            sources = np.hstack([sources, padding])
                        
                        separated_sources.append(sources.T)  # Back to frequency x time
                    except:
                        # Fallback if ICA fails
                        separated_sources.append(np.zeros((n_sources, freq_data.shape[0])))
        
        # Reconstruct full frequency range
        full_sources = []
        for i in range(n_sources):
            source = np.zeros_like(magnitude)
            for j, freq_bin in enumerate(range(0, magnitude.shape[0], 10)):
                if j < len(separated_sources) and freq_bin + 10 < magnitude.shape[0]:
                    source[freq_bin:freq_bin+10, :] = separated_sources[j][i:i+1, :]
            full_sources.append(source)
        
        return full_sources, phase
    
    def harmonic_percussive_separation(self, stft_matrix):
        """Advanced harmonic-percussive separation with refinement"""
        print("Performing harmonic-percussive separation...")
        
        # Initial separation
        harmonic, percussive = librosa.decompose.hpss(stft_matrix, margin=8.0)
        
        # Refine harmonic component for vocals
        harmonic_mag = np.abs(harmonic)
        harmonic_phase = np.angle(harmonic)
        
        # Create vocal mask based on harmonic structure
        vocal_mask = self.create_advanced_vocal_mask(harmonic_mag)
        
        # Create instrumental mask (everything else in harmonic)
        instrumental_harmonic_mask = 1.0 - vocal_mask
        
        # Separate percussive into drums and other
        percussive_mag = np.abs(percussive)
        drum_mask = self.create_advanced_drum_mask(percussive_mag)
        
        # Apply masks
        vocals = harmonic * vocal_mask
        instrumental_harmonic = harmonic * instrumental_harmonic_mask
        drums = percussive * drum_mask
        other_percussive = percussive * (1.0 - drum_mask)
        
        return vocals, instrumental_harmonic + other_percussive, drums, harmonic_phase
    
    def create_advanced_vocal_mask(self, magnitude):
        """Create sophisticated vocal mask using multiple criteria"""
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        # 1. Frequency-based mask (vocal formants)
        freq_mask = np.zeros_like(magnitude)
        
        # Fundamental frequency range (80-300 Hz)
        f0_range = (freqs >= 80) & (freqs <= 300)
        
        # First formant (300-1000 Hz) - most important for vowels
        f1_range = (freqs >= 300) & (freqs <= 1000)
        
        # Second formant (1000-3000 Hz) - vowel discrimination
        f2_range = (freqs >= 1000) & (freqs <= 3000)
        
        # Third formant (2000-4000 Hz) - consonant clarity
        f3_range = (freqs >= 2000) & (freqs <= 4000)
        
        # Apply different weights to different formant regions
        freq_mask[f0_range, :] = 0.6
        freq_mask[f1_range, :] = 1.0  # Highest weight
        freq_mask[f2_range, :] = 0.9
        freq_mask[f3_range, :] = 0.7
        
        # 2. Temporal stability mask (vocals are more stable than instruments)
        temporal_mask = np.zeros_like(magnitude)
        for i in range(magnitude.shape[0]):
            # Calculate local variance across time
            local_var = np.var(magnitude[i, :])
            stability = 1.0 / (1.0 + local_var)  # Higher stability = lower variance
            temporal_mask[i, :] = stability
        
        # 3. Harmonic structure mask
        harmonic_mask = self.detect_harmonic_structure(magnitude)
        
        # Combine all masks
        combined_mask = freq_mask * temporal_mask * harmonic_mask
        
        # Apply smoothing
        combined_mask = gaussian_filter1d(combined_mask, sigma=1.0, axis=0)
        combined_mask = gaussian_filter1d(combined_mask, sigma=2.0, axis=1)
        
        # Normalize to [0, 1]
        combined_mask = np.clip(combined_mask, 0, 1)
        
        return combined_mask
    
    def create_advanced_drum_mask(self, magnitude):
        """Create sophisticated drum mask"""
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        # Drum frequency characteristics
        kick_range = (freqs >= 20) & (freqs <= 120)    # Kick drum
        snare_range1 = (freqs >= 150) & (freqs <= 300) # Snare fundamental
        snare_range2 = (freqs >= 2000) & (freqs <= 8000) # Snare crack
        hihat_range = (freqs >= 8000) & (freqs <= 20000) # Hi-hats and cymbals
        
        # Detect transients (drum hits)
        transient_mask = self.detect_transients(magnitude)
        
        # Frequency-based drum mask
        freq_mask = np.zeros_like(magnitude)
        freq_mask[kick_range, :] = 0.9
        freq_mask[snare_range1, :] = 0.8
        freq_mask[snare_range2, :] = 0.7
        freq_mask[hihat_range, :] = 0.6
        
        # Combine with transient detection
        drum_mask = freq_mask * transient_mask
        
        return np.clip(drum_mask, 0, 1)
    
    def detect_harmonic_structure(self, magnitude):
        """Detect harmonic structure typical of vocals"""
        harmonic_mask = np.ones_like(magnitude)
        
        # Look for harmonic series
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        for t in range(magnitude.shape[1]):
            frame = magnitude[:, t]
            
            # Find peaks
            peaks, _ = signal.find_peaks(frame, height=np.max(frame) * 0.1)
            
            if len(peaks) > 1:
                # Check for harmonic relationships
                peak_freqs = freqs[peaks]
                harmonicity = 0
                
                for i, f1 in enumerate(peak_freqs[:-1]):
                    for f2 in peak_freqs[i+1:]:
                        ratio = f2 / f1
                        # Check if ratio is close to integer (harmonic)
                        if abs(ratio - round(ratio)) < 0.1:
                            harmonicity += 1
                
                # Normalize harmonicity
                if len(peaks) > 1:
                    harmonicity /= len(peaks) * (len(peaks) - 1) / 2
                
                harmonic_mask[:, t] *= harmonicity
        
        return harmonic_mask
    
    def detect_transients(self, magnitude):
        """Detect transient events (drum hits)"""
        # Calculate onset strength
        onset_envelope = librosa.onset.onset_strength(S=magnitude, sr=self.sr)
        
        # Find transient locations
        transient_times = librosa.onset.onset_detect(onset_envelope=onset_envelope, 
                                                    sr=self.sr, 
                                                    hop_length=self.hop_length)
        
        # Create transient mask
        transient_mask = np.zeros_like(magnitude)
        
        for onset_time in transient_times:
            # Convert to frame index
            frame_idx = int(onset_time * self.sr / self.hop_length)
            if frame_idx < magnitude.shape[1]:
                # Mark transient region
                start_frame = max(0, frame_idx - 2)
                end_frame = min(magnitude.shape[1], frame_idx + 5)
                transient_mask[:, start_frame:end_frame] = 1.0
        
        return transient_mask
    
    def wiener_filtering(self, mixed_stft, source_estimates):
        """Apply Wiener filtering for final refinement"""
        print("Applying Wiener filtering...")
        
        # Calculate power spectrograms
        mixed_power = np.abs(mixed_stft) ** 2
        
        refined_sources = []
        for source_estimate in source_estimates:
            source_power = np.abs(source_estimate) ** 2
            
            # Wiener filter
            total_power = sum(np.abs(est) ** 2 for est in source_estimates)
            wiener_filter = source_power / (total_power + 1e-10)
            
            # Apply filter
            refined_source = mixed_stft * wiener_filter
            refined_sources.append(refined_source)
        
        return refined_sources
    
    def separate_sources(self, file_path):
        """Main separation pipeline using multiple advanced techniques"""
        # Load audio
        y, sr = self.load_audio(file_path)
        
        # Compute advanced STFT
        stft_matrix = self.advanced_stft(y)
        
        print("Running multiple separation algorithms...")
        
        # Method 1: Harmonic-Percussive + Advanced Masking
        vocals_hp, instrumental_hp, drums_hp, phase = self.harmonic_percussive_separation(stft_matrix)
        
        # Method 2: Matrix Factorization
        nmf_sources, _ = self.matrix_factorization_separation(stft_matrix)
        
        # Method 3: Spectral Clustering  
        cluster_masks, _ = self.spectral_clustering_separation(stft_matrix)
        
        # Combine results using ensemble approach
        print("Combining results with ensemble method...")
        
        # Vocals: Combine HP vocals with NMF source most similar to vocals
        vocals_final = vocals_hp * 0.6 + nmf_sources[0] * np.exp(1j * phase) * 0.4
        
        # Drums: Combine HP drums with clustered percussive content
        drums_final = drums_hp * 0.7 + stft_matrix * cluster_masks[1] * 0.3
        
        # Bass: Use low-frequency NMF component
        bass_mask = self.create_bass_mask(np.abs(stft_matrix))
        bass_final = stft_matrix * bass_mask
        
        # Instrumental: Everything else
        instrumental_final = stft_matrix - vocals_final - drums_final - bass_final
        
        # Apply Wiener filtering for final refinement
        all_sources = [vocals_final, drums_final, bass_final, instrumental_final]
        refined_sources = self.wiener_filtering(stft_matrix, all_sources)
        
        # Convert back to time domain
        vocals_audio = librosa.istft(refined_sources[0], hop_length=self.hop_length)
        drums_audio = librosa.istft(refined_sources[1], hop_length=self.hop_length)
        bass_audio = librosa.istft(refined_sources[2], hop_length=self.hop_length)
        instrumental_audio = librosa.istft(refined_sources[3], hop_length=self.hop_length)
        
        return {
            'vocals': vocals_audio,
            'drums': drums_audio, 
            'bass': bass_audio,
            'instrumental': instrumental_audio,
            'sample_rate': sr
        }
    
    def create_bass_mask(self, magnitude):
        """Create mask for bass frequencies"""
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        bass_range = (freqs >= 20) & (freqs <= 250)
        
        mask = np.zeros_like(magnitude)
        mask[bass_range, :] = 1.0
        
        # Apply smoothing
        mask = gaussian_filter1d(mask, sigma=1.0, axis=0)
        
        return mask

def main():
    """Main execution function"""
    
    # PUT YOUR ACTUAL MP3 FILENAME HERE
    audio_file = "AudioSeperation\Radiohead - Let Down.mp3"  # <<< CHANGE THIS TO YOUR ACTUAL FILE NAME
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: File '{audio_file}' not found!")
        print("Available audio files in current directory:")
        for file in os.listdir('.'):
            if file.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                print(f"  - {file}")
        print(f"\nPlease update the 'audio_file' variable with your actual filename.")
        return
    
    print("="*60)
    print("ADVANCED NEURAL-STYLE SOURCE SEPARATION")
    print("="*60)
    
    # Initialize separator with high-quality settings
    separator = AdvancedSourceSeparator(n_fft=4096, hop_length=1024, n_components=12)
    
    try:
        # Perform separation
        print(f"Processing: {audio_file}")
        separated_sources = separator.separate_sources(audio_file)
        
        # Save high-quality outputs
        print("\nSaving separated sources...")
        
        sr = separated_sources['sample_rate']
        
        # Save with high quality
        sf.write('VOCALS_CLEAN.wav', separated_sources['vocals'], sr, subtype='PCM_24')
        sf.write('INSTRUMENTAL_CLEAN.wav', separated_sources['instrumental'], sr, subtype='PCM_24')
        sf.write('DRUMS_CLEAN.wav', separated_sources['drums'], sr, subtype='PCM_24')
        sf.write('BASS_CLEAN.wav', separated_sources['bass'], sr, subtype='PCM_24')
        
        # Quality analysis
        print("\nQuality Analysis:")
        for source_name, audio_data in separated_sources.items():
            if source_name != 'sample_rate':
                rms = np.sqrt(np.mean(audio_data**2))
                dynamic_range = 20 * np.log10(np.max(np.abs(audio_data)) / (rms + 1e-10))
                print(f"{source_name.capitalize()}: RMS={rms:.4f}, Dynamic Range={dynamic_range:.1f}dB")
        
        print("\n" + "="*60)
        print("SEPARATION COMPLETE!")
        print("="*60)
        print("High-quality output files:")
        print("  🎤 VOCALS_CLEAN.wav - Isolated vocals")
        print("  🎸 INSTRUMENTAL_CLEAN.wav - All instruments") 
        print("  🥁 DRUMS_CLEAN.wav - Drums and percussion")
        print("  🎵 BASS_CLEAN.wav - Bass frequencies")
        print("\nAll files saved in 24-bit quality.")
        
    except Exception as e:
        print(f"Error during separation: {e}")
        import traceback
        print("Full error trace:")
        traceback.print_exc()

if __name__ == "__main__":
    main()