import uvicorn
import uuid
import json
import time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Security, BackgroundTasks, Form
from fastapi.responses import Response, JSONResponse

# Import our modules
import config
from utils import logger, cleanup_local_folder, cleanup_file
import comfy_interface
import workflow_logic

app = FastAPI()

# --- SECURITY ---
async def get_api_key(api_key_header: str = Security(config.API_KEY_HEADER)):
    if api_key_header == config.API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Invalid API Key")

# --- HEALTH CHECK ROUTE ---
@app.get("/checkhealth")
def check_health():
    return {"status": "online"}

# --- HELPER ---
def parse_job_id(composite_id: str):
    """Splits the composite ID back into prompt_id and session_id."""
    try:
        prompt_id, session_id = composite_id.split("::")
        return prompt_id, session_id
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Job ID format. Expected 'prompt_id::session_id'.")

# --- ROUTE: SUBMIT ---
@app.post("/process")
def submit_job(
    target_image: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    # Optional Parameters
    remover_steps: Optional[int] = Form(None),
    remover_strength: Optional[float] = Form(None),
    remover_cfg: Optional[float] = Form(None),
    transfer_steps: Optional[int] = Form(None),
    transfer_cfg: Optional[float] = Form(None),
    transfer_control_strength: Optional[float] = Form(None),
    transfer_adapter_strength: Optional[float] = Form(None),
    api_key: str = Security(get_api_key)):

    logger.info("⚡ New Job Submission Received")

    # 1. Health Check
    if not comfy_interface.check_health():
        time.sleep(5)
        if not comfy_interface.check_health():
            raise HTTPException(status_code=503, detail="ComfyUI is initializing or failed")

    session_id = str(uuid.uuid4())
    logger.info(f"🆔 Session Created: {session_id}")

    try:
        # 2. Upload Images
        logger.info("📤 Uploading images...")
        t_bytes = target_image.file.read()
        r_bytes = reference_image.file.read()
        
        target_path = comfy_interface.upload_image(t_bytes, target_image.filename, session_id)
        ref_path = comfy_interface.upload_image(r_bytes, reference_image.filename, session_id)

        # 3. Prepare Workflow
        try:
            with open(config.WORKFLOW_FILE, "r") as f:
                workflow = json.load(f)
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="Workflow JSON not found")

        mappings = workflow_logic.get_dynamic_mappings(workflow)
        
        options = {
            "remover_steps": remover_steps,
            "remover_strength": remover_strength,
            "remover_cfg": remover_cfg,
            "transfer_steps": transfer_steps,
            "transfer_cfg": transfer_cfg,
            "transfer_control_strength": transfer_control_strength,
            "transfer_adapter_strength": transfer_adapter_strength,
        }
        
        workflow = workflow_logic.prepare_workflow(workflow, mappings, target_path, ref_path, options)

        # 4. Submit to Queue
        prompt_id = comfy_interface.queue_prompt(workflow, session_id)
        
        # 5. Return Composite ID
        # We combine prompt_id and session_id so we can recover the session context (folder paths) later.
        job_id = f"{prompt_id}::{session_id}"
        
        return {"job_id": job_id, "status": "queued"}

    except Exception as e:
        logger.error(f"🔥 Submission Error: {str(e)}")
        # Clean up immediately if submission fails
        cleanup_local_folder(session_id, config.INPUT_DIR_ROOT)
        raise HTTPException(status_code=500, detail=str(e))

# --- ROUTE: STATUS ---
@app.get("/status/{job_id}")
def get_job_status(job_id: str, api_key: str = Security(get_api_key)):
    prompt_id, session_id = parse_job_id(job_id)
    
    # 1. Check History (Completed?)
    history = comfy_interface.get_history(prompt_id)
    if history:
        return {"job_id": job_id, "status": "completed"}
    
    # 2. Check Queue (Pending or Running?)
    queue_info = comfy_interface.get_queue_info()
    if not queue_info:
        # Could not fetch queue, but also not in history. Ambiguous or ComfyUI down.
        raise HTTPException(status_code=503, detail="Unable to fetch queue status")

    # Check Running
    # item structure: [number, prompt_id, ...]
    for item in queue_info.get('queue_running', []):
        if item[1] == prompt_id:
            return {"job_id": job_id, "status": "processing"}
            
    # Check Pending
    for i, item in enumerate(queue_info.get('queue_pending', [])):
        if item[1] == prompt_id:
            return {
                "job_id": job_id, 
                "status": "queued", 
                "queue_position": i + 1
            }

    # If not in history, running, or pending -> Unknown/Failed
    return {"job_id": job_id, "status": "unknown", "detail": "Job not found in queue or history"}

# --- ROUTE: RESULT ---
@app.get("/result/{job_id}")
def get_job_result(
    background_tasks: BackgroundTasks,
    job_id: str, 
    api_key: str = Security(get_api_key)
):
    prompt_id, session_id = parse_job_id(job_id)
    
    # 1. Check History
    history = comfy_interface.get_history(prompt_id)
    if not history:
        # It might be still processing or failed
        raise HTTPException(status_code=404, detail="Job not completed or not found")
        
    # 2. Extract Output
    try:
        outputs = history['outputs']
        # We need to find the save node. 
        # Since we don't have 'mappings' here easily without re-parsing workflow, 
        # we look for any node that produced 'images' with type 'output'.
        
        img_meta = None
        for node_id, node_output in outputs.items():
            if 'images' in node_output:
                for img in node_output['images']:
                    if img.get('type') == 'output':
                        img_meta = img
                        break
            if img_meta:
                break
        
        if not img_meta:
             raise HTTPException(status_code=500, detail="No output image found in history")

        logger.info(f"🖼️  Downloading result for {job_id}: {img_meta['filename']}")
        
        img_data = comfy_interface.get_image(
            img_meta['filename'], 
            img_meta['subfolder'], 
            img_meta['type']
        )
        
        if img_data:
            # Schedule Cleanup
            background_tasks.add_task(cleanup_local_folder, session_id, config.INPUT_DIR_ROOT)
            background_tasks.add_task(cleanup_file, img_meta['filename'], img_meta['subfolder'], config.OUTPUT_DIR_ROOT)
            
            return Response(content=img_data, media_type="image/png")
            
        raise HTTPException(status_code=500, detail="Failed to retrieve image file")

    except Exception as e:
        logger.error(f"Error retrieving result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info(f"🟢 API Server starting on port {config.API_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=config.API_PORT)
