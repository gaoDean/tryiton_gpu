import logging
import os
import shutil

# --- LOGGING SETUP ---
def setup_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', 
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = setup_logger("stablehair_api")

# --- FILE CLEANUP ---
def cleanup_local_folder(folder_name, root_path):
    """Clean up the input subfolder after processing."""
    try:
        full_path = os.path.join(root_path, folder_name)
        if os.path.exists(full_path):
            shutil.rmtree(full_path)
            logger.info(f"🧹 Cleaned up input folder: {folder_name}")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

def cleanup_file(filename, subfolder, root_path):
    """Clean up the output file after processing."""
    try:
        full_path = os.path.join(root_path, subfolder, filename)
        if os.path.exists(full_path):
            os.remove(full_path)
            logger.info(f"🧹 Cleaned up output file: {filename}")
    except Exception as e:
        logger.error(f"Cleanup file failed: {e}")