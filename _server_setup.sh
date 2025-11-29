#!/bin/bash

# --- 1. SYSTEM & BUILD TOOLS ---
echo "--- Updating System and Installing Build Tools ---"
apt-get update
# libgl1 and libglib are strictly required for OpenCV
apt-get install -y wget git build-essential libgl1-mesa-glx libglib2.0-0 python3-venv python3-dev

# --- 2. WORKSPACE SETUP & CLONING ---
cd /workspace
git clone git@github.com:gaoDean/tryiton_gpu.git
cd tryiton_gpu

# define paths for easy invocation
/venv/main/bin/pip install uv
PIP_CMD="/venv/main/bin/uv pip install --python /venv/main/bin/python"
PYTHON_CMD="/venv/main/bin/python"


# --- 3. INSTALL DEPENDENCIES ---

# Enable HF Transfer for faster model downloads later
export HF_HUB_ENABLE_HF_TRANSFER=1
$PIP_CMD hf-transfer

/venv/main/bin/hf download zhengchong/FastFit-MR-1024 --local-dir Models/FastFit-MR-1024
/venv/main/bin/hf download zhengchong/Human-Toolkit --local-dir Models/Human-Toolkit

echo "--- Installing Detectron2 (Git) ---"
# Detectron2 is the most fragile dependency; we install it via git to ensure CUDA compatibility
# We install this BEFORE requirements.txt to prevent conflicts
$PIP_CMD install 'git+https://github.com/facebookresearch/detectron2.git'

echo "--- Installing Requirements ---"
if [ -f "requirements.txt" ]; then
    $PIP_CMD install -r requirements.txt
else
    echo "WARNING: requirements.txt not found"
fi

# --- 4. FINAL CONFIGURATION ---

# Ensure the inference script is executable if present
if [ -f "inference.py" ]; then
    chmod +x inference.py
fi

# Create directory for inputs if it doesn't exist
mkdir -p inputs

echo "----------------------------------------------------"
echo "SETUP COMPLETE"
echo "----------------------------------------------------"