import os
import librosa
import soundfile as sf
import numpy as np
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.audio import AudioFile
from demucs.pretrained import get_model

class AudioProcessor:
    def __init__(self):
        # Initialize Demucs for vocal separation
        self.demucs_model = get_model(name='htdemucs').cpu()
        print("Demucs model loaded successfully")
    
    def process(self, input_path, action, params):
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(os.path.dirname(input_path), f'processed_{name}.wav')

        try:
            # Load audio file
            y, sr = librosa.load(input_path, sr=None)

            # Extract parameters with safe defaults and type conversion
            speed = float(params.get('speed', 1.0))
            pitch = int(params.get('pitch', 0))
            reverb = str(params.get('reverb', 'false')).lower() == 'true'
            echo = str(params.get('echo', 'false')).lower() == 'true'
            separate = str(params.get('separate', 'false')).lower() == 'true'

            print(f"Params -> speed: {speed}, pitch: {pitch}, reverb: {reverb}, echo: {echo}, separate: {separate}")

            # Apply speed and pitch
            if speed != 1.0 or pitch != 0:
                y = self.change_speed_stereo(y, sr, speed, pitch)

            # Apply reverb
            if reverb:
                y = self.add_reverb(y, sr)

            # Apply echo
            if echo:
                y = self.add_echo(y, sr)

            # Vocal separation overrides all other effects
            if separate:
                vocals_path = output_path.replace('.wav', '_vocals.wav')
                self.separate_vocals(input_path, vocals_path)
                return f'processed_{name}_vocals.wav'

            # Save processed file
            sf.write(output_path, y, sr)
            return f'processed_{name}.wav'

        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            raise


    def change_speed(self, y, sr, speed, pitch_shift):
        # Convert to mono if stereo
        if y.ndim > 1:
            y = librosa.to_mono(y)
        
        # Time stretch
        if speed != 1.0:
            y = librosa.effects.time_stretch(y, rate=speed)
        
        # Pitch shift
        if pitch_shift != 0:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)

        return y
    def change_speed_stereo(self, y, sr, speed, pitch_shift):
        print(f"[DEBUG] Stereo shape: {y.shape}")
        if y.ndim == 1:
            return self.change_speed(y, sr, speed, pitch_shift)
        
        # Stretch each channel
        left = librosa.effects.time_stretch(y[0], rate=float(speed))
        right = librosa.effects.time_stretch(y[1], rate=float(speed))

        # Match lengths
        min_len = min(len(left), len(right))
        left, right = left[:min_len], right[:min_len]

        # Pitch shift if needed
        if float(pitch_shift) != 0:
            left = librosa.effects.pitch_shift(left, sr=sr, n_steps=float(pitch_shift))
            right = librosa.effects.pitch_shift(right, sr=sr, n_steps=float(pitch_shift))

        return np.vstack((left, right))

    
    def add_reverb(self, y, sr):
        # Simple reverb effect
        reverberated = np.copy(y)
        for _ in range(5):  # Number of echoes
            echo = np.roll(y, int(0.1 * sr))  # 100ms delay
            echo *= 0.5  # Decay factor
            reverberated += echo
        return reverberated
    
    def add_echo(self, y, sr):
        # Simple echo effect
        echo = np.roll(y, int(0.3 * sr))  # 300ms delay
        echo *= 0.7  # Echo volume
        return y + echo
    
    def separate_vocals(self, input_path, output_path):
        """Separate vocals using Demucs"""
        try:
            # Load audio file with Demucs
            wav = AudioFile(input_path).read(streams=0, 
                                          samplerate=self.demucs_model.samplerate, 
                                          channels=self.demucs_model.audio_channels)
            
            # Apply separation
            sources = apply_model(self.demucs_model, 
                                wav[None], 
                                device='cpu', 
                                split=True, 
                                overlap=0.25, 
                                progress=True)
            
            # Demucs returns: [drums, bass, other, vocals]
            vocals = sources[0, 3].cpu().numpy()
            
            # Save vocals
            sf.write(output_path, vocals.T, self.demucs_model.samplerate)
            return True
            
        except Exception as e:
            print(f"Error in vocal separation: {str(e)}")
            raise

    def convert_format(self, input_path, output_path, format='wav'):
        """Convert between audio formats using torchaudio"""
        waveform, sample_rate = torchaudio.load(input_path)
        torchaudio.save(output_path, waveform, sample_rate, format=format)