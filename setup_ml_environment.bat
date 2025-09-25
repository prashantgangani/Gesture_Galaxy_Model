@echo off
echo Setting up ML environment for Sign Language Recognition...

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install numpy first (required for mediapipe)
pip install numpy==1.24.3

REM Install opencv-python
pip install opencv-python==4.8.1.78

REM Install scikit-learn
pip install scikit-learn==1.3.2

REM Install Pillow
pip install Pillow==10.0.1

REM Try to install mediapipe with specific flags to avoid common errors
echo Installing MediaPipe (this might take a few minutes)...
pip install mediapipe==0.10.8 --no-deps
pip install protobuf>=3.11,<4

REM If mediapipe fails, try alternative installation
if errorlevel 1 (
    echo MediaPipe installation failed, trying alternative method...
    pip install --upgrade pip setuptools wheel
    pip install mediapipe==0.10.8 --no-cache-dir
)

echo.
echo ML environment setup complete!
echo To activate the environment, run: venv\Scripts\activate.bat
echo To test the installation, run: python -c "import mediapipe; print('MediaPipe installed successfully!')"
pause

