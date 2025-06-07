from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from audio_processor import AudioProcessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav'}

processor = AudioProcessor()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the file and return the processed version
        action = request.form.get('action', '')
        params = {
            'speed': float(request.form.get('speed', 1.0)),
            'pitch': float(request.form.get('pitch', 0)),
            'reverb': request.form.get('reverb', 'false') == 'true',
            'echo': request.form.get('echo', 'false') == 'true',
            'separate': request.form.get('separate', 'false') == 'true'
        }
        
        output_filename = processor.process(filepath, action, params)
        return jsonify({'filename': output_filename})
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)