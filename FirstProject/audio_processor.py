import os
import librosa
import soundfile as sf
import numpy as np
import torch
import torchaudio
from demucs.pretrained import get_model_from_args
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
            
            # Apply effects based on action
            if action == 'speed':
                y = self.change_speed(y, sr, params['speed'], params['pitch'])
            if params['reverb']:
                y = self.add_reverb(y, sr)
            if params['echo']:
                y = self.add_echo(y, sr)
            if params['separate']:
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
        # Time stretching
        y_stretched = librosa.effects.time_stretch(y, rate=speed)
        
        # Pitch shifting if needed
        if pitch_shift != 0:
            y_stretched = librosa.effects.pitch_shift(y_stretched, sr=sr, n_steps=pitch_shift)
        
        return y_stretched
    
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