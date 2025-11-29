#!/bin/bash

# --- SYSTEM & BUILD TOOLS ---

apt-get update
apt-get install -y wget git build-essential supervisor libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config libgl1-mesa-glx libglib2.0-0 unzip

# Install uv for faster pip installs
/venv/main/bin/pip install uv
PIP_INSTALL="/venv/main/bin/uv pip install --python /venv/main/bin/python"

# --- CLONE REPOSITORIES ---

cd /workspace

if [ ! -d "tryiton_gpu" ]; then
    git clone https://github.com/gaoDean/tryiton_gpu.git
    mv /workspace/tryiton_gpu/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
fi

if [ ! -d "ComfyUI" ]; then
    git clone https://github.com/comfyanonymous/ComfyUI.git
fi

mkdir -p /workspace/ComfyUI/custom_nodes
cd /workspace/ComfyUI/custom_nodes

if [ ! -d "ComfyUI_essentials" ]; then
    git clone https://github.com/cubiq/ComfyUI_essentials.git
fi

if [ ! -d "FastFit" ]; then
    echo "Downloading FastFit Release..."
    wget -O FastFit.zip https://github.com/Zheng-Chong/FastFit/releases/download/comfyui/FastFit.zip
    unzip FastFit.zip
    rm FastFit.zip
fi

if [ ! -d "ComfyUI-Safety-Checker" ]; then
    git clone https://github.com/shabri-arrahim/ComfyUI-Safety-Checker.git
fi

# --- INSTALL ALL PYTHON DEPENDENCIES ---

$PIP_INSTALL easy-dwpose --no-dependencies
$PIP_INSTALL \
    -r /workspace/tryiton_gpu/requirements.txt \
    -r /workspace/ComfyUI/requirements.txt \
    -r /workspace/ComfyUI/custom_nodes/FastFit/requirements.txt \
    -r /workspace/ComfyUI/custom_nodes/ComfyUI-Safety-Checker/requirements.txt \
    -r /workspace/ComfyUI/custom_nodes/ComfyUI_essentials/requirements.txt \
    huggingface_hub hf-transfer av

export HF_HUB_ENABLE_HF_TRANSFER=1

# --- DOWNLOAD MODELS ---

mkdir -p /workspace/ComfyUI/models/checkpoints/FastFit
cd /workspace/ComfyUI/models/checkpoints/FastFit

if [ ! -d "FastFit-MR-1024" ]; then
    echo "Downloading FastFit-MR Diffusers Model (Full Repo)..."
    /venv/main/bin/hf download zhengchong/FastFit-MR-1024 --local-dir FastFit-MR-1024
fi

# --- INSTALL NGROK ---
if ! command -v ngrok &> /dev/null; then
    curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
    echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list
    apt-get update
    apt-get install -y ngrok jq
fi

if [ -z "$NGROK_AUTH" ]; then
    echo "WARNING: NGROK_AUTH environment variable is not set."
else
    ngrok config add-authtoken $NGROK_AUTH
fi

# --- CONFIGURE SUPERVISOR ---

service supervisor start
supervisorctl reread
supervisorctl update
supervisorctl start api_server ngrok comfyui

echo "Deployment Complete"
echo "Server is accessible at: https://unintrigued-epigamic-rowan.ngrok-free.dev"
