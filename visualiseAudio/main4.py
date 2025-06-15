import os
import numpy as np
import torch
import soundfile as sf
import streamlit as st
from openunmix import predict
from pydub import AudioSegment
import io

# Set page config
st.set_page_config(
    page_title="Audio Separation Studio",
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .stAudio {
        width: 100%;
    }
    .stButton>button {
        width: 100%;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        border: 1px solid #4CAF50;
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
        border: 1px solid #45a049;
    }
    .stSelectbox>div>div>div {
        color: #4CAF50;
    }
    .stFileUploader>section>div>button {
        color: white;
        background-color: #4CAF50;
    }
    .stFileUploader>section>div>button:hover {
        background-color: #45a049;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def separate_audio(filename, targets=['vocals', 'drums', 'bass', 'other']):
    """Separate audio into different tracks using OpenUnmix"""
    try:
        audio, sr = sf.read(filename, always_2d=True)
        audio_tensor = torch.from_numpy(audio.T).float()
        audio_tensor = audio_tensor.unsqueeze(0)
        
        estimates = predict.separate(
            audio=audio_tensor,
            rate=sr,
            targets=targets
        )
        return estimates, sr
    except Exception as e:
        st.error(f"Error during audio separation: {str(e)}")
        return None, None

def save_audio(output_dir, filename, estimates, sr):
    """Save separated tracks to files"""
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}
    for target, estimate in estimates.items():
        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(filename))[0]}_{target}.wav")
        estimate_np = estimate.squeeze(0).detach().cpu().numpy().T
        sf.write(output_file, estimate_np, sr)
        saved_files[target] = output_file
    return saved_files

def main():
    st.title("🎶 Audio Separation Studio")
    st.markdown("""
    Separate your audio tracks (vocals, drums, bass, other) with this powerful tool.
    Upload your audio file and customize the processing below.
    """)
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload Audio File", type=['mp3', 'wav', 'ogg', 'flac'])
        
        st.subheader("Separation Options")
        targets = st.multiselect(
            "Select tracks to separate",
            ['vocals', 'drums', 'bass', 'other'],
            default=['vocals', 'drums', 'bass', 'other']
        )
    
    # Main content area
    if uploaded_file is not None:
        # Display original audio
        st.subheader("Original Audio")
        st.audio(uploaded_file)
        
        # Process button
        if st.button("Separate Tracks"):
            with st.spinner("Processing audio... This may take a few minutes depending on file size."):
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Save uploaded file temporarily
                temp_dir = "temp_audio"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                progress_bar.progress(20)
                
                # Separate audio
                estimates, sr = separate_audio(temp_path, targets=targets)
                
                if estimates is not None:
                    progress_bar.progress(60)
                    
                    # Save separated tracks
                    saved_files = save_audio(temp_dir, uploaded_file.name, estimates, sr)
                    progress_bar.progress(80)
                    
                    # Display and allow download of each track
                    st.subheader("Separated Tracks")
                    cols = st.columns(2)
                    
                    for i, (target, file_path) in enumerate(saved_files.items()):
                        with cols[i % 2]:
                            st.markdown(f"**{target.capitalize()}**")
                            
                            # Create in-memory file for preview and download
                            with open(file_path, "rb") as f:
                                audio_bytes = f.read()
                            
                            # Display audio player
                            st.audio(audio_bytes)
                            
                            # Download button
                            st.download_button(
                                label=f"Download {target}",
                                data=audio_bytes,
                                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_{target}.wav",
                                mime="audio/wav"
                            )
                    
                    progress_bar.progress(100)
                    st.success("Audio processing complete!")
                    
                    # Clean up temporary files
                    try:
                        os.remove(temp_path)
                        for file_path in saved_files.values():
                            os.remove(file_path)
                    except:
                        pass
                else:
                    st.error("Failed to process the audio file.")

if __name__ == '__main__':
    main()