import requests
import json
import time
import subprocess
import mimetypes
from utils import logger
import config

def check_health():
    """Checks health. If dead, triggers supervisor restart."""
    try:
        requests.get(config.COMFY_URL, timeout=2)
        return True
    except:
        logger.warning("⚠️ ComfyUI unreachable. Restarting via Supervisor...")
        subprocess.run(["supervisorctl", "restart", "comfyui"])
        time.sleep(10) 
        return False

def upload_image(image_data, filename, subfolder):
    """Uploads via ComfyUI API."""
    try:
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            mime_type = 'image/png' # Fallback

        files = {'image': (filename, image_data, mime_type)}
        data = {"overwrite": "false", "subfolder": subfolder, "type": "input"}
        
        response = requests.post(f"{config.COMFY_URL}/upload/image", files=files, data=data)
        response.raise_for_status()
        resp_json = response.json()
        
        # Format the path string Comfy expects (folder/file)
        if resp_json['subfolder']:
            return f"{resp_json['subfolder']}/{resp_json['name']}"
        return resp_json['name']
        
    except Exception as e:
        logger.error(f"Upload failed for {filename}: {e}")
        raise e

def get_image(filename, subfolder, folder_type):
    """Downloads via ComfyUI API."""
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    response = requests.get(f"{config.COMFY_URL}/view", params=params)
    if response.status_code == 200:
        return response.content
    return None

def queue_prompt(workflow, client_id):
    """
    Submits prompt to ComfyUI and returns the prompt_id (job_id).
    Non-blocking.
    """
    payload = {"prompt": workflow, "client_id": client_id}
    try:
        response = requests.post(f"{config.COMFY_URL}/prompt", json=payload)
        response.raise_for_status()
        resp_json = response.json()
        prompt_id = resp_json['prompt_id']
        logger.info(f"🚀 Prompt Queued. ID: {prompt_id}")
        return prompt_id
    except Exception as e:
        logger.error(f"Failed to queue prompt: {e}")
        raise e

def get_queue_info():
    """
    Retrieves the current queue state from ComfyUI.
    Returns a dict with 'queue_running' and 'queue_pending' lists.
    """
    try:
        response = requests.get(f"{config.COMFY_URL}/queue")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch queue info: {e}")
        return None

def get_history(prompt_id):
    """
    Retrieves history for a specific prompt_id to check for completion/outputs.
    Returns the history dict for that ID if found, else None.
    """
    try:
        response = requests.get(f"{config.COMFY_URL}/history/{prompt_id}")
        if response.status_code == 200:
            history = response.json()
            if prompt_id in history:
                return history[prompt_id]
        return None
    except Exception as e:
        logger.error(f"Failed to fetch history for {prompt_id}: {e}")
        return None
