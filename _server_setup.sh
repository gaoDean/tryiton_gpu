#!/bin/bash

# --- 1. SYSTEM & BUILD TOOLS ---
echo "--- Updating System and Installing Build Tools ---"
apt-get update
# libgl1 and libglib are strictly required for OpenCV
apt-get install -y wget git build-essential unzip supervisor

# --- 2. NGROK SETUP ---
if [ -z "$NGROK_AUTH" ]; then
    echo "WARNING: NGROK_AUTH environment variable not set. Ngrok may not work."
else
    echo "--- Setting up Ngrok ---"
    wget -q https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip
    unzip -o ngrok-v3-stable-linux-amd64.zip
    mv ngrok /usr/local/bin/ngrok
    rm ngrok-v3-stable-linux-amd64.zip
    ngrok config add-authtoken $NGROK_AUTH
fi

# --- 3. WORKSPACE SETUP & CLONING ---
cd /workspace
git clone https://github.com/gaoDean/tryiton_gpu.git 
cd tryiton_gpu
mv supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# define paths for easy invocation
/venv/main/bin/pip install uv
PIP_CMD="/venv/main/bin/uv pip install --python /venv/main/bin/python"
PYTHON_CMD="/venv/main/bin/python"


# --- 4. INSTALL DEPENDENCIES ---

# Enable HF Transfer for faster model downloads later
export HF_HUB_ENABLE_HF_TRANSFER=1
$PIP_CMD hf-transfer huggingface_hub

/venv/main/bin/hf download zhengchong/FastFit-MR-1024 --local-dir Models/FastFit-MR-1024
/venv/main/bin/hf download zhengchong/Human-Toolkit --local-dir Models/Human-Toolkit

$PIP_CMD -r requirements.txt
/venv/main/bin/pip install easy-dwpose --no-dependencies

# --- 5. FINAL CONFIGURATION ---

# Ensure the inference script is executable if present
if [ -f "inference.py" ]; then
    chmod +x inference.py
fi

echo "--- Starting and configuring Supervisord services ---"
service supervisor start
supervisorctl reread
supervisorctl update
supervisorctl start ngrok server

echo "----------------------------------------------------"
echo "SETUP COMPLETE"
echo "----------------------------------------------------"
