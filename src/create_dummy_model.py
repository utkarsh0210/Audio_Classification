import torch
import torchaudio
import numpy as np
import os

# Constants for audio processing (same as in the original script)
SAMPLE_RATE = 22050
DURATION = 4
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FIXED_AUDIO_LENGTH = SAMPLE_RATE * DURATION

# Class labels
CLASS_LABELS = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", 
    "drilling", "engine_idling", "gun_shot", "jackhammer", 
    "siren", "street_music"
]

# Create a dummy model for testing
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

# Create a dummy model and save it
input_shape = (1, N_MELS, 173)  # (channels, height, width)
model = SimpleCNN(num_classes=len(CLASS_LABELS), input_shape=input_shape)

# Create the models directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)

# Save the model
torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'models', 'urban_sound_cnn_pytorch.pth'))

print("Dummy model created and saved successfully!")
