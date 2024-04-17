from flask import Flask, request, jsonify, make_response
import os
import io
import zipfile
import librosa
import numpy as np
import soundfile as sf
import logging
from vocal_remover.lib import spec_utils
from inference_changed import MainProcessor, Separator

app = Flask(__name__)

@app.route('/separate_audio', methods=['POST'])
def separate_audio():
    try:
        if 'file' not in request.files:
            logging.error('No file part.')
            return jsonify({'error': 'No file part.'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            logging.error('No selected file.')
            return jsonify({'error': 'No selected file.'}), 400
        
        # Save the audio file temporarily
        filename = 'temp_audio.wav'
        file.save(filename)
        
        # Load audio file
        X, sr = librosa.load(filename, sr=22050, mono=False, dtype=np.float32, res_type='kaiser_fast')
        
        if X.ndim == 1:
            # mono to stereo
            X = np.asarray([X, X])
        
        X_spec = spec_utils.wave_to_spectrogram(X, hop_length=512, n_fft=2048)
        
        # Initialize and set the model instance before creating Separator
        model_path = 'vocal_remover/models/baseline.pth'

device = torch.device('cpu')
model = nets.CascadedNet(2048, 1024, 32, 128, True)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to(device)
        
        sp = Separator(
            model=model,
            device=device,
            batchsize=4,
            cropsize=256,
        )
        
        y_spec, v_spec = sp.separate(X_spec)
        
        instruments_wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=512)
        vocals_wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=512)
        
        instruments_file = io.BytesIO()
        vocals_file = io.BytesIO()
        
        sf.write(instruments_file, instruments_wave.T, sr, format='wav')
        sf.write(vocals_file, vocals_wave.T, sr, format='wav')
        
        response = make_response()
        response.headers["Content-Disposition"] = "attachment; filename=separated_files.zip"
        response.headers["Content-type"] = "application/zip"
        
        with io.BytesIO() as zip_buffer:
            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                zip_file.writestr('instruments.wav', instruments_file.getvalue())
                zip_file.writestr('vocals.wav', vocals_file.getvalue())
            
            response.data = zip_buffer.getvalue()
        
        # Delete the temporary audio file
        os.remove(filename)
        
        return response
    
    except Exception as e:
        logging.error(f'Error occurred: {str(e)}')
        return jsonify({'error': str(e)}), 500
