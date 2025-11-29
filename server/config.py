import os
from fastapi.security import APIKeyHeader

# --- NETWORK SETTINGS ---
API_PORT = 9000
COMFY_HOST = "127.0.0.1"
COMFY_PORT = "8188"
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"

# --- PATHS ---
# Adjust these if your folder structure changes
INPUT_DIR_ROOT = "/workspace/ComfyUI/input"
OUTPUT_DIR_ROOT = "/workspace/ComfyUI/output"
WORKFLOW_FILE = "workflow_api.json"

# --- SECURITY ---
API_KEY = "midgets"
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)