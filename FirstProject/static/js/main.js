document.addEventListener('DOMContentLoaded', function() {
    // Initialize WaveSurfer
    const wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#4a4a4a',
        progressColor: '#0d6efd',
        cursorColor: '#0d6efd',
        barWidth: 2,
        barRadius: 3,
        cursorWidth: 1,
        height: 100,
        barGap: 2
    });
    
    const audioPlayer = document.getElementById('audioPlayer');
    let currentAudio = null;
    
    // Update display values
    document.getElementById('speedControl').addEventListener('input', function() {
        document.getElementById('speedValue').textContent = this.value;
    });
    
    document.getElementById('pitchControl').addEventListener('input', function() {
        document.getElementById('pitchValue').textContent = this.value;
    });
    
    // Upload and process audio
    document.getElementById('uploadBtn').addEventListener('click', function() {
        const fileInput = document.getElementById('audioFile');
        if (fileInput.files.length === 0) {
            alert('Please select an audio file first');
            return;
        }
        
        processAudio(fileInput.files[0]);
    });
    
    // Process with current settings
    document.getElementById('processBtn').addEventListener('click', function() {
        if (!currentAudio) {
            alert('Please upload an audio file first');
            return;
        }
        
        processAudio(currentAudio);
    });
    
    function processAudio(file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('action', 'process');
        formData.append('speed', document.getElementById('speedControl').value);
        formData.append('pitch', document.getElementById('pitchControl').value);
        formData.append('reverb', document.getElementById('reverbControl').checked);
        formData.append('echo', document.getElementById('echoControl').checked);
        formData.append('separate', document.getElementById('separateControl').checked);
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }
            
            // Update audio player
            currentAudio = file;
            const audioUrl = `/processed/${data.filename}`;
            audioPlayer.src = audioUrl;
            
            // Update waveform
            wavesurfer.load(audioUrl);
            
            // Play audio
            audioPlayer.play();
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while processing the audio');
        });
    }
    
    // Play/pause with WaveSurfer
    wavesurfer.on('interaction', () => {
        wavesurfer.play();
    });
});