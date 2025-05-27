# Environmental Sound Classifier Flask App - README

## Overview
This Flask application integrates an environmental sound classifier model to provide a web interface for audio classification. Users can upload audio files through the web interface, and the application will classify the sound into one of several predefined categories.

## Project Structure
```
sound_classifier_app/
├── requirements.txt        # Python dependencies
├── src/
│   ├── main.py             # Flask application entry point
│   ├── create_dummy_model.py  # Script to create a test model
│   ├── models/             # Directory for model files
│   │   └── urban_sound_cnn_pytorch.pth  # Trained model weights
│   ├── routes/             # API routes (for future expansion)
│   ├── static/             # Static assets (CSS, JS, images)
│   ├── templates/          # HTML templates
│   │   └── index.html      # Main webpage
│   └── uploads/            # Temporary directory for uploaded files
└── venv/                   # Virtual environment (not included in zip)
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation
1. Unzip the provided `sound_classifier_app.zip` file
2. Navigate to the extracted directory:
   ```
   cd sound_classifier_app
   ```
3. Create a virtual environment:
   ```
   python -m venv venv
   ```
4. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```
5. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application
1. Ensure you're in the project directory with the virtual environment activated
2. Start the Flask server:
   ```
   cd src
   python -m flask --app main run
   ```
3. Access the application in your web browser at:
   ```
   http://127.0.0.1:5000
   ```

## Using the Application
1. Open the web interface in your browser
2. Drag and drop an audio file onto the upload area (or click to browse)
3. Wait for the file to upload and process
4. View the classification results, including:
   - Predicted sound class
   - Confidence score
   - Probability distribution across all classes

## Model Information
- The application uses a CNN-based model for environmental sound classification
- The model is trained to recognize 10 different sound classes:
  - air_conditioner
  - car_horn
  - children_playing
  - dog_bark
  - drilling
  - engine_idling
  - gun_shot
  - jackhammer
  - siren
  - street_music

## Customization
- To use your own trained model, replace the file at `src/models/urban_sound_cnn_pytorch.pth`
- Ensure your model follows the same architecture as defined in `main.py`
- If your model uses different class labels, update the `CLASS_LABELS` list in `main.py`

## Troubleshooting
- If you encounter "Model not loaded" errors, ensure the model file exists in the correct location
- For audio processing errors, verify that the uploaded file is in a supported audio format
- If the server fails to start, check that all dependencies are correctly installed
