#!/bin/bash
# Installation automatique Fall Detection sur Raspberry Pi 5

set -e

echo "🍓 Fall Detection Installation for Raspberry Pi 5"
echo "=================================================="

# 1. Update system
echo "[1/7] Updating system..."
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-pip python3-dev git libatlas-base-dev
sudo apt install -y libwebp6 libtiff5 libharfbuzz0b libwebpmux3

# Note: libjasper-dev removed in Bullseye - not needed for fall detection

# 2. Install Python 3.11
echo "[2/7] Checking Python version..."
if ! python3 --version | grep -q "3.11"; then
    echo "Installing Python 3.11..."
    sudo apt install -y python3.11 python3.11-venv python3.11-dev
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
fi
python3 --version

# 3. Clone repository
echo "[3/7] Cloning repository..."
cd /home/pi
if [ ! -d "fall" ]; then
    git clone https://github.com/ayman2218/fall.git
else
    echo "Repository already exists, pulling latest..."
    cd fall && git pull && cd ..
fi

# 4. Create virtual environment
echo "[4/7] Creating virtual environment..."
cd /home/pi/fall
python3.11 -m venv fall_env
source fall_env/bin/activate

# 5. Install dependencies
echo "[5/7] Installing Python packages (this may take 5-10 minutes)..."
pip install --upgrade pip setuptools wheel
# Pin protobuf to <4 — mediapipe 0.10.x is incompatible with protobuf 4.x+
pip install "protobuf>=3.20.3,<4.0.0"
pip install numpy opencv-python mediapipe

# Optional: Uncomment for PyTorch/TensorFlow (slower on Pi)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# pip install tensorflow

echo "✓ Python packages installed"

# 6. Create directory structure
echo "[6/7] Creating directory structure..."
mkdir -p models

echo "✓ Directory structure created"

# 7. Final check
echo "[7/7] Final verification..."
python3 -c "import cv2; import mediapipe; print('✓ All imports successful')"

echo ""
echo "=================================================="
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Download models (pose_landmarker_lite.task, etc.) into models/"
echo "2. Connect USB camera"
echo "3. Run: source /home/pi/fall/fall_env/bin/activate"
echo "4. Run: python3 /home/pi/fall/fall_detector_pi.py"
echo ""
echo "For auto-start, follow instructions in RASPBERRY_PI_5_SETUP.md"
echo "=================================================="
