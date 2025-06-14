import os
import numpy as np
import torch
from openunmix import predict
import soundfile as sf

def separate_audio(filename, targets=['vocals', 'drums', 'other', 'bass']):
    # Load the audio file first
    audio, sr = sf.read(filename, always_2d=True)
    
    # Convert to torch tensor and add batch dimension
    audio_tensor = torch.from_numpy(audio.T).float()  # (channels, samples)
    audio_tensor = audio_tensor.unsqueeze(0)  # (batch, channels, samples)
    
    # Run separation
    estimates = predict.separate(
        audio=audio_tensor,
        rate=sr,
        targets=targets
    )
    return estimates, sr

def save_audio(output_dir, filename, estimates, sr):
    os.makedirs(output_dir, exist_ok=True)
    for target, estimate in estimates.items():
        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(filename))[0]}_{target}.wav")
        
        # Convert to numpy and handle dimensions properly
        estimate_np = estimate.squeeze(0).detach().cpu().numpy()  # Remove batch dim
        estimate_np = estimate_np.T  # (samples, channels)
        
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