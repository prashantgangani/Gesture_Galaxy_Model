#!/bin/bash
echo "Setting up ML environment for Sign Language Recognition..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip first
python -m pip install --upgrade pip

# Install numpy first (required for mediapipe)
pip install numpy==1.24.3

# Install opencv-python
pip install opencv-python==4.8.1.78

# Install scikit-learn
pip install scikit-learn==1.3.2

# Install Pillow
pip install Pillow==10.0.1

# Try to install mediapipe with specific flags to avoid common errors
echo "Installing MediaPipe (this might take a few minutes)..."
pip install mediapipe==0.10.8 --no-deps
pip install protobuf>=3.11,<4

# If mediapipe fails, try alternative installation
if [ $? -ne 0 ]; then
    echo "MediaPipe installation failed, trying alternative method..."
    pip install --upgrade pip setuptools wheel
    pip install mediapipe==0.10.8 --no-cache-dir
fi

echo ""
echo "ML environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To test the installation, run: python -c \"import mediapipe; print('MediaPipe installed successfully!')\""

