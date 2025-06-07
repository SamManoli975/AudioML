import tempfile
import os
from pydub import AudioSegment
from pydub.playback import play

# Use your own custom temp dir
custom_tempdir = os.path.abspath("temp_audio")
os.makedirs(custom_tempdir, exist_ok=True)
tempfile.tempdir = custom_tempdir  # override default tempdir

# Load and play MP3
sound = AudioSegment.from_file("audio.mp3", format="mp3")
play(sound)
