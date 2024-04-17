from flask import Flask, request, jsonify
import os
from inference_changed import MainProcessor

app = Flask(_name_)

@app.route('/separate_audio', methods=['POST'])
def separate_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the audio file temporarily
        filename = 'temp_audio.wav'
        file.save(filename)
        
        # Process the audio file
        MainProcessor.mainu(filename)
        
        # Delete the temporary audio file
        os.remove(filename)
        
        # Return success message
        return jsonify({'message': 'Audio separation completed successfully'})
    else:
        return jsonify({'error': 'Failed to process audio'})

if _name_ == '_main_':
    app.run(host="0.0.0.0", port=6463,debug=True)