import os
import numpy as np
import torch
from openunmix import predict
import soundfile as sf
import librosa

def separate_audio(filename, targets=['vocals', 'drums', 'melody']):
    # Load the audio file first
    audio, sr = sf.read(filename, always_2d=True)
    
    # Convert to torch tensor and add batch dimension
    audio_tensor = torch.from_numpy(audio.T).float()  # (channels, samples)
    audio_tensor = audio_tensor.unsqueeze(0)  # (batch, channels, samples)
    
    # Run separation for standard targets (excluding melody)
    standard_targets = [t for t in targets if t != 'melody']
    estimates = predict.separate(
        audio=audio_tensor,
        rate=sr,
        targets=standard_targets
    )
    
    # Add melody extraction if requested
    if 'melody' in targets:
        estimates['melody'] = extract_melody(audio, sr)
    
    return estimates, sr

def extract_melody(audio, sr):
    # Convert to mono for melody extraction
    mono_audio = librosa.to_mono(audio.T)  # Transpose to (samples, channels) first
    
    # Extract melody frequencies using YIN algorithm
    melody_freq = librosa.yin(
        mono_audio,
        fmin=librosa.note_to_hz('C2'),  # ~65 Hz
        fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
        sr=sr,
        frame_length=2048,
        hop_length=512
    )
    
    # Generate sine wave from frequencies
    times = np.arange(len(mono_audio)) / sr
    melody_audio = np.sin(2 * np.pi * melody_freq * times)
    
    # Convert back to stereo and normalize
    melody_audio = np.vstack([melody_audio, melody_audio])  # Make stereo
    melody_audio = librosa.util.normalize(melody_audio) * 0.7  # 70% volume
    
    return melody_audio.T  # Return in (samples, channels) format

def save_audio(output_dir, filename, estimates, sr):
    os.makedirs(output_dir, exist_ok=True)
    for target, estimate in estimates.items():
        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(filename))[0]}_{target}.wav")
        
        if isinstance(estimate, torch.Tensor):
            # Convert tensor to numpy for standard stems
            estimate_np = estimate.squeeze(0).detach().cpu().numpy().T
        else:
            # Melody is already numpy array in correct format
            estimate_np = estimate
        
        sf.write(output_file, estimate_np, sr)
        print(f"✅ Saved {target} track to {output_file}")

def process_audio(filename, output_dir='output'):
    # Separate the audio
    estimates, sr = separate_audio(filename)
    
    # Save the separated tracks
    save_audio(output_dir, filename, estimates, sr)

if __name__ == '__main__':
    filename = 'visualiseAudio/letdown.mp3'
    process_audio(filename)