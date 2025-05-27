import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # DON'T CHANGE THIS !!!

from flask import Flask, render_template, request, jsonify
import torch
import torchaudio
import numpy as np
import librosa
import io
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(os.path.join(os.path.dirname(__file__), app.config['UPLOAD_FOLDER']), exist_ok=True)

# Constants for audio processing (same as in the original script)
SAMPLE_RATE = 22050
DURATION = 4
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FIXED_AUDIO_LENGTH = SAMPLE_RATE * DURATION

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels (to be updated with actual labels)
CLASS_LABELS = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", 
    "drilling", "engine_idling", "gun_shot", "jackhammer", 
    "siren", "street_music"
]

# Model definition (same as in the original script)
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes, input_shape):
        super(SimpleCNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape)
            dummy_output = self.conv_layers(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).size(1)

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(self.flattened_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Audio processing functions
def load_audio_file(file_path, target_sr=SAMPLE_RATE):
    try:
        waveform, sr = torchaudio.load(file_path)
        return waveform, sr
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None

def preprocess_audio(waveform, sr, target_sr=SAMPLE_RATE, target_length=FIXED_AUDIO_LENGTH, trim_silence=True):
    # Ensure waveform is a torch tensor
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()

    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    # Convert to mono (if stereo)
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Squeeze out the channel dimension if it's 1, to make it (samples,)
    if waveform.ndim > 1 and waveform.shape[0] == 1:
         waveform = waveform.squeeze(0)

    # Convert to numpy for librosa operations like trim
    waveform_np = waveform.numpy()

    # Trim silence
    if trim_silence:
        waveform_trimmed, _ = librosa.effects.trim(waveform_np, top_db=30)
        if len(waveform_trimmed) == 0:
            print(f"Warning: Trimming resulted in empty audio. Using original before trim.")
        else:
            waveform_np = waveform_trimmed

    # Convert back to tensor
    waveform = torch.from_numpy(waveform_np).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Normalize audio
    if torch.max(torch.abs(waveform)) > 0:
        waveform = waveform / torch.max(torch.abs(waveform))
    else:
        waveform = torch.zeros_like(waveform)

    # Pad or truncate to fixed length
    current_length = waveform.shape[1]
    if current_length > target_length:
        waveform = waveform[:, :target_length]
    elif current_length < target_length:
        padding = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, padding), mode='constant')

    return waveform.squeeze(0)

def generate_mel_spectrogram(waveform, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    mel_spectrogram = mel_spectrogram_transform(waveform)

    # Convert to decibels (log scale)
    log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)(mel_spectrogram)
    
    return log_mel_spectrogram.squeeze(0)

# Global model variable
model = None

def load_model(model_path):
    global model
    try:
        # Determine input shape for the model
        # This is a placeholder - in a real app, you'd need to know the exact input shape
        input_shape = (1, N_MELS, 173)  # (channels, height, width)
        
        # Initialize model
        model = SimpleCNN(num_classes=len(CLASS_LABELS), input_shape=input_shape)
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        print(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def predict(audio_path):
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Load and preprocess audio
        waveform, sr = load_audio_file(audio_path)
        if waveform is None:
            return {"error": "Failed to load audio file"}
        
        processed_waveform = preprocess_audio(waveform, sr)
        mel_spec = generate_mel_spectrogram(processed_waveform)
        
        # Add batch and channel dimensions
        mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time_frames)
        
        # Make prediction
        with torch.no_grad():
            mel_spec = mel_spec.to(DEVICE)
            outputs = model(mel_spec)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
        
        # Get class probabilities
        class_probs = {}
        for i, prob in enumerate(probabilities.cpu().numpy()):
            class_probs[CLASS_LABELS[i]] = float(prob)
        
        return {
            "predicted_class": CLASS_LABELS[predicted_class],
            "confidence": float(probabilities[predicted_class]),
            "probabilities": class_probs
        }
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(os.path.dirname(__file__), app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Make prediction
        result = predict(file_path)
        
        # Clean up the uploaded file
        os.remove(file_path)
        
        return jsonify(result)

@app.route('/model-info')
def model_info():
    if model is None:
        return jsonify({"status": "Model not loaded"})
    
    return jsonify({
        "status": "Model loaded",
        "device": str(DEVICE),
        "classes": CLASS_LABELS
    })

if __name__ == '__main__':
    # Try to load model if available
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'urban_sound_cnn_pytorch.pth')
    if os.path.exists(model_path):
        load_model(model_path)
    else:
        print(f"Warning: Model file not found at {model_path}")
        print("The app will run, but predictions won't work until a model is loaded.")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
