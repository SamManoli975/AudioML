import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
import os
import warnings
import subprocess
import sys
warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check if FFmpeg is available
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# Install pydub only if FFmpeg is available
ffmpeg_available = check_ffmpeg()
if ffmpeg_available:
    from pydub import AudioSegment
    print("FFmpeg is available. MP3 support enabled.")
else:
    print("FFmpeg not found. MP3 files will be handled with librosa (may have limited support).")

class VoiceEncoder(nn.Module):
    """
    Voice Encoder Network: Extracts speaker embeddings from audio
    """
    def __init__(self, hidden_size=256, num_layers=3):
        super(VoiceEncoder, self).__init__()
        
        # CNN for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Projection to speaker embedding
        self.projection = nn.Linear(hidden_size * 2, 256)
        
    def forward(self, x):
        # x shape: (batch_size, 1, seq_len)
        x = self.conv_layers(x)  # (batch_size, 256, seq_len/8)
        x = x.transpose(1, 2)    # (batch_size, seq_len/8, 256)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len/8, hidden_size*2)
        
        # Average pooling over time
        pooled = torch.mean(lstm_out, dim=1)  # (batch_size, hidden_size*2)
        
        # Project to speaker embedding
        embedding = self.projection(pooled)  # (batch_size, 256)
        embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalization
        
        return embedding

class SimpleTTS(nn.Module):
    """
    Simplified TTS model that generates mel-spectrograms from text and speaker embeddings
    """
    def __init__(self, vocab_size=100, mel_dim=80, hidden_dim=256):
        super(SimpleTTS, self).__init__()
        
        # Text embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Decoder (simplified)
        self.decoder = nn.LSTM(
            input_size=hidden_dim * 2 + 256,  # encoder output + speaker embedding
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Linear projection to mel-spectrogram
        self.mel_projection = nn.Linear(hidden_dim, mel_dim)
        
    def forward(self, text, speaker_embedding):
        # Text embedding
        embedded = self.embedding(text)  # (batch_size, seq_len, hidden_dim)
        
        # Encoder
        encoder_out, _ = self.encoder(embedded)  # (batch_size, seq_len, hidden_dim*2)
        
        # Expand speaker embedding to match encoder output
        spk_emb_expanded = speaker_embedding.unsqueeze(1).expand(-1, encoder_out.size(1), -1)
        
        # Concatenate encoder output with speaker embedding
        decoder_input = torch.cat([encoder_out, spk_emb_expanded], dim=2)
        
        # Decoder
        decoder_out, _ = self.decoder(decoder_input)
        
        # Mel-spectrogram projection
        mel_output = self.mel_projection(decoder_out)
        
        return mel_output.transpose(1, 2)  # (batch_size, mel_dim, seq_len)

class GriffinLimVocoder:
    """
    Griffin-Lim algorithm for converting mel-spectrograms to audio
    This is a simpler alternative to neural vocoders
    """
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80, n_iter=50):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_iter = n_iter
        
        # Mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate, 
            n_fft=n_fft, 
            n_mels=n_mels
        )
        
        # Inverse mel filterbank
        self.inv_mel_basis = np.linalg.pinv(self.mel_basis)
    
    def mel_to_audio(self, mel):
        """
        Convert mel-spectrogram to audio using Griffin-Lim algorithm
        
        Args:
            mel (numpy.array): Mel-spectrogram (n_mels, time)
            
        Returns:
            numpy.array: Reconstructed audio
        """
        # Inverse mel scaling
        spec = np.dot(self.inv_mel_basis, mel)
        spec = np.maximum(1e-10, spec)  # Avoid log(0)
        
        # Griffin-Lim algorithm
        angles = np.exp(2j * np.pi * np.random.rand(*spec.shape))
        for i in range(self.n_iter):
            # Reconstruct complex spectrum
            complex_spec = spec * angles
            
            # Inverse STFT
            audio = librosa.istft(complex_spec, hop_length=self.hop_length)
            
            # Forward STFT
            stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            
            # Update phase
            angles = np.exp(1j * np.angle(stft))
        
        # Final reconstruction
        complex_spec = spec * angles
        audio = librosa.istft(complex_spec, hop_length=self.hop_length)
        
        return audio

class VoiceCloningSystem:
    def __init__(self):
        """
        Initialize the complete voice cloning system
        """
        self.sample_rate = 22050
        self.hop_length = 256
        self.n_fft = 1024
        self.n_mels = 80
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.voice_encoder = VoiceEncoder().to(self.device)
        self.tts_model = SimpleTTS().to(self.device)
        self.vocoder = GriffinLimVocoder()
        
        print("Voice cloning system initialized successfully!")
    
    def load_audio(self, audio_path):
        """
        Load audio file (supports MP3, WAV, and other formats)
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            numpy.array: Audio data
            int: Sample rate
        """
        try:
            # Get absolute path to help with debugging
            abs_path = os.path.abspath(audio_path)
            print(f"Trying to load: {abs_path}")
            
            # Check if file exists
            if not os.path.exists(abs_path):
                print(f"Error: File does not exist at path: {abs_path}")
                print(f"Current working directory: {os.getcwd()}")
                print("Files in current directory:")
                for f in os.listdir('.'):
                    print(f"  {f}")
                return None, None
            
            # Check file extension
            file_ext = os.path.splitext(audio_path)[1].lower()
            print(f"File extension: {file_ext}")
            
            if file_ext == '.mp3' and ffmpeg_available:
                # Load MP3 using pydub
                print("Loading MP3 file with pydub...")
                audio = AudioSegment.from_mp3(abs_path)
                audio = audio.set_frame_rate(self.sample_rate)
                audio = audio.set_channels(1)  # Convert to mono
                audio_data = np.array(audio.get_array_of_samples()).astype(np.float32)
                audio_data = audio_data / (2**15)  # Normalize to [-1, 1]
                print(f"MP3 loaded successfully, length: {len(audio_data)} samples")
                return audio_data, self.sample_rate
            else:
                # Load using librosa (works for MP3 if ffmpeg is available system-wide)
                print(f"Loading {file_ext} file using librosa...")
                audio, sr = librosa.load(abs_path, sr=self.sample_rate, mono=True)
                print(f"Audio loaded successfully, length: {len(audio)} samples")
                return audio, sr
                
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None, None
    
    def preprocess_audio(self, audio_path, target_duration=5.0):
        """
        Preprocess audio file for voice cloning
        
        Args:
            audio_path (str): Path to audio file
            target_duration (float): Target duration in seconds
            
        Returns:
            torch.Tensor: Preprocessed audio tensor
        """
        # Load audio
        print(f"Preprocessing audio from: {audio_path}")
        audio, sr = self.load_audio(audio_path)
        if audio is None:
            raise ValueError(f"Could not load audio from {audio_path}")
        
        # Trim silence
        print("Trimming silence...")
        audio, _ = librosa.effects.trim(audio, top_db=20)
        print(f"After trimming: {len(audio)} samples")
        
        # Normalize to target duration
        target_samples = int(target_duration * self.sample_rate)
        if len(audio) > target_samples:
            # Take the middle segment
            start = (len(audio) - target_samples) // 2
            audio = audio[start:start + target_samples]
            print(f"Trimmed to target duration: {len(audio)} samples")
        else:
            # Pad with zeros
            padding = target_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
            print(f"Padded to target duration: {len(audio)} samples")
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
        
        return audio_tensor
    
    def extract_speaker_embedding(self, audio_tensor):
        """
        Extract speaker embedding from audio
        
        Args:
            audio_tensor (torch.Tensor): Audio tensor
            
        Returns:
            torch.Tensor: Speaker embedding
        """
        print("Extracting speaker embedding...")
        self.voice_encoder.eval()
        with torch.no_grad():
            # Add batch dimension if needed
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Extract embedding
            embedding = self.voice_encoder(audio_tensor.unsqueeze(1))
        
        print("Speaker embedding extracted successfully")
        return embedding
    
    def text_to_mel(self, text, speaker_embedding):
        """
        Convert text to mel-spectrogram using the TTS model
        
        Args:
            text (str): Input text
            speaker_embedding (torch.Tensor): Speaker embedding
            
        Returns:
            numpy.array: Generated mel-spectrogram
        """
        print("Converting text to mel-spectrogram...")
        self.tts_model.eval()
        
        # Convert text to token IDs (simplified)
        text = text.lower()
        tokens = [min(ord(c) % 100, 99) for c in text if c.isalnum() or c.isspace()]  # Simple character-based tokenization
        if not tokens:
            tokens = [0]  # Default token if text is empty
        
        print(f"Text tokens: {tokens}")
        tokens = torch.LongTensor(tokens).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Generate mel-spectrogram
            mel_output = self.tts_model(tokens, speaker_embedding)
        
        # Convert to numpy and denormalize
        mel = mel_output.squeeze(0).cpu().numpy()
        mel = np.exp(mel)  # Assuming log-mel was output
        
        print("Mel-spectrogram generated successfully")
        return mel
    
    def mel_to_audio(self, mel):
        """
        Convert mel-spectrogram to audio using the Griffin-Lim vocoder
        
        Args:
            mel (numpy.array): Mel-spectrogram
            
        Returns:
            numpy.array: Generated audio
        """
        print("Converting mel-spectrogram to audio...")
        audio = self.vocoder.mel_to_audio(mel)
        print(f"Audio generated with {len(audio)} samples")
        return audio
    
    def clone_voice(self, source_audio_path, text, output_path="cloned_voice.wav"):
        """
        Complete voice cloning pipeline
        
        Args:
            source_audio_path (str): Path to source audio for voice cloning
            text (str): Text to synthesize
            output_path (str): Path to save generated audio
            
        Returns:
            bool: True if successful
        """
        print("=" * 50)
        print("Starting voice cloning process...")
        print("=" * 50)
        
        try:
            # Step 1: Preprocess source audio
            print("\n1. Preprocessing source audio...")
            audio_tensor = self.preprocess_audio(source_audio_path)
            
            # Step 2: Extract speaker embedding
            print("\n2. Extracting speaker embedding...")
            speaker_embedding = self.extract_speaker_embedding(audio_tensor)
            
            # Step 3: Generate mel-spectrogram from text
            print("\n3. Generating mel-spectrogram from text...")
            mel = self.text_to_mel(text, speaker_embedding)
            
            # Step 4: Convert mel-spectrogram to audio
            print("\n4. Converting mel-spectrogram to audio...")
            audio = self.mel_to_audio(mel)
            
            # Step 5: Save generated audio
            print("\n5. Saving generated audio...")
            sf.write(output_path, audio, self.sample_rate)
            
            print("=" * 50)
            print(f"Voice cloning completed! Output saved to {output_path}")
            print("=" * 50)
            return True
            
        except Exception as e:
            print(f"Error during voice cloning: {e}")
            import traceback
            traceback.print_exc()
            return False

def find_audio_files(directory="."):
    """
    Find all audio files in the current directory
    """
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac']
    audio_files = []
    
    print("Searching for audio files in current directory...")
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(file)
            print(f"Found audio file: {file}")
    
    return audio_files

def install_ffmpeg_windows():
    """
    Try to install FFmpeg on Windows
    """
    print("Attempting to install FFmpeg...")
    try:
        # Try to install using conda if available
        try:
            subprocess.run([sys.executable, "-m", "conda", "install", "-c", "conda-forge", "ffmpeg", "-y"], 
                         capture_output=True, check=True)
            print("FFmpeg installed successfully via conda")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Try to install using pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "ffmpeg-python"], 
                         capture_output=True, check=True)
            print("FFmpeg Python bindings installed successfully")
            return True
        except subprocess.CalledProcessError:
            pass
            
        print("Could not automatically install FFmpeg.")
        print("Please install FFmpeg manually:")
        print("1. Download from: https://www.gyan.dev/ffmpeg/builds/")
        print("2. Extract the ZIP file")
        print("3. Add the 'bin' folder to your system PATH")
        return False
        
    except Exception as e:
        print(f"Error installing FFmpeg: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Check for FFmpeg and try to install if not available
    if not ffmpeg_available and sys.platform.startswith('win'):
        install_ffmpeg_windows()
        # Check again after attempted installation
        ffmpeg_available = check_ffmpeg()
    
    # Initialize the voice cloning system
    vc_system = VoiceCloningSystem()
    
    # Find all audio files in the current directory
    audio_files = find_audio_files()
    
    if not audio_files:
        print("No audio files found in the current directory!")
        print("Please place your audio file in the same directory as this script.")
        print("Supported formats: MP3, WAV, OGG, FLAC, M4A, AAC")
    else:
        # Use the first audio file found
        source_audio = audio_files[0]
        print(f"Using audio file: {source_audio}")
        
        # Clone voice
        text_to_speak = "Hello, this is my cloned voice speaking!"
        success = vc_system.clone_voice(source_audio, text=text_to_speak, output_path="cloned_voice.wav")
        
        if success:
            print("Voice cloning process completed successfully!")
        else:
            print("Voice cloning process failed!")
            
            # If failed due to MP3 issues, suggest converting to WAV
            if source_audio.lower().endswith('.mp3') and not ffmpeg_available:
                print("\nMP3 conversion tip:")
                print("You can convert your MP3 to WAV using online tools like:")
                print("1. https://online-audio-converter.com/")
                print("2. https://cloudconvert.com/mp3-to-wav")
                print("Then try again with the WAV file.")