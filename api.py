from flask import Flask, request, jsonify, make_response, send_file
import argparse
import os
import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from vocal_remover.lib import dataset
from vocal_remover.lib import nets
from vocal_remover.lib import spec_utils
from vocal_remover.lib import utils
import io
import zipfile
app = Flask(__name__)


# Load model
model_path = 'vocal_remover/models/baseline.pth'

device = torch.device('cpu')
model = nets.CascadedNet(2048, 1024, 32, 128, True)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to(device)

class Separator(object):
    def __init__(self, model, device=None, batchsize=1, cropsize=256, postprocess=False):
        self.model = model
        self.offset = model.offset
        self.device = device
        self.batchsize = batchsize
        self.cropsize = cropsize
        self.postprocess = postprocess

    def _postprocess(self, X_spec, mask):
        if self.postprocess:
            mask_mag = np.abs(mask)
            mask_mag = spec_utils.merge_artifacts(mask_mag)
            mask = mask_mag * np.exp(1.j * np.angle(mask))

        y_spec = X_spec * mask
        v_spec = X_spec - y_spec

        return y_spec, v_spec

    def _separate(self, X_spec_pad, roi_size):
        X_dataset = []
        patches = (X_spec_pad.shape[2] - 2 * self.offset) // roi_size
        for i in range(patches):
            start = i * roi_size
            X_spec_crop = X_spec_pad[:, :, start:start + self.cropsize]
            X_dataset.append(X_spec_crop)

        X_dataset = np.asarray(X_dataset)
        with torch.no_grad():
            mask_list = []
           
            for i in tqdm(range(0, patches, self.batchsize)):
                X_batch = X_dataset[i: i + self.batchsize]
                X_batch = torch.from_numpy(X_batch).to(self.device)

                mask = self.model.predict_mask(X_batch)

                mask = mask.detach().cpu().numpy()
                mask = np.concatenate(mask, axis=2)
                mask_list.append(mask)

            mask = np.concatenate(mask_list, axis=2)

        return mask

    def separate(self, X_spec):
        n_frame = X_spec.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_spec_pad = np.pad(X_spec, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_spec_pad /= np.abs(X_spec).max()

        mask = self._separate(X_spec_pad, roi_size)
        mask = mask[:, :, :n_frame]

        y_spec, v_spec = self._postprocess(X_spec, mask)

        return y_spec, v_spec

    def separate_tta(self, X_spec):
        n_frame = X_spec.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_spec_pad = np.pad(X_spec, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_spec_pad /= X_spec_pad.max()

        mask = self._separate(X_spec_pad, roi_size)

        pad_l += roi_size // 2
        pad_r += roi_size // 2
        X_spec_pad = np.pad(X_spec, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_spec_pad /= X_spec_pad.max()

        mask_tta = self._separate(X_spec_pad, roi_size)
        mask_tta = mask_tta[:, :, roi_size // 2:]
        mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5

        y_spec, v_spec = self._postprocess(X_spec, mask)

        return y_spec, v_spec

import logging

logging.basicConfig(level=logging.DEBUG)

@app.route('/separate', methods=['POST'])
def separate_audio():
    try:
    
        if 'audio_file' not in request.files:
            logging.error('No audio file provided.')
            return jsonify({'error': 'No audio file provided.'}), 400

        audio_file = request.files['audio_file']

        audio_filename = 'uploaded_audio.wav'
        audio_file.save(audio_filename)
        # Load audio file
        if not os.path.exists(audio_filename):
            logging.error('Audio file does not exist.')
            return jsonify({'error': 'Audio file does not exist.'}), 400

        X, sr = librosa.load(audio_filename, sr=22050, mono=False, dtype=np.float32, res_type='kaiser_fast')

        if X.ndim == 1:
            # mono to stereo
            X = np.asarray([X, X])

        X_spec = spec_utils.wave_to_spectrogram(X, hop_length=512, n_fft=2048)

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

        return response




    except Exception as e:
        logging.error(f'Error occurred: {str(e)}')
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6400, debug=True)