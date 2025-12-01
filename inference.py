import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional

# --- Path Setup ---
sys.path.append(os.getcwd())

from module.pipeline_fastfit import FastFitPipeline
from parse_utils import DWposeDetector, DensePose, SCHP, cloth_agnostic_mask
from parse_utils.automasker import part_mask_of, LIP_MAPPING, ATR_MAPPING

# --- Configuration ---
PERSON_SIZE = (768, 1024)
CLOTHING_SIZE = (768, 1024)
ACCESSORY_SIZE = (384, 512)

def center_crop_to_aspect_ratio(img: Image.Image, target_ratio: float) -> Image.Image:
    width, height = img.size
    current_ratio = width / height
    
    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        new_height = height
        left = (width - new_width) // 2
        top = 0
    else:
        new_width = width
        new_height = int(width / target_ratio)
        left = 0
        top = (height - new_height) // 2
        
    return img.crop((left, top, left + new_width, top + new_height))

class FastFitEngine:
    def __init__(self, device="cuda", mixed_precision="bf16"):
        self.device = device
        self.base_model_path = "Models/FastFit-MR-1024"
        self.util_model_path = "Models/Human-Toolkit"
        
        # Validation: Check if models exist
        if not os.path.exists(self.base_model_path) or not os.path.exists(self.util_model_path):
            raise FileNotFoundError(
                "\nERROR: Models not found!\n"
                "Please run 'python setup_models.py' first to download the necessary weights."
            )

        print(f"--- Initializing FastFit on {self.device.upper()} ---")
        self._load_pipeline(mixed_precision)

    def _load_pipeline(self, mixed_precision):
        print("Loading Preprocessors...")
        self.dwpose = DWposeDetector(
            pretrained_model_name_or_path=os.path.join(self.util_model_path, "DWPose"), 
            device=self.device
        )
        self.densepose = DensePose(
            model_path=os.path.join(self.util_model_path, "DensePose"), 
            device=self.device
        )
        self.schp_lip = SCHP(
            ckpt_path=os.path.join(self.util_model_path, "SCHP", "schp-lip.pth"), 
            device=self.device
        )
        self.schp_atr = SCHP(
            ckpt_path=os.path.join(self.util_model_path, "SCHP", "schp-atr.pth"), 
            device=self.device
        )
        
        print("Loading Diffusion Pipeline...")
        self.pipeline = FastFitPipeline(
            base_model_path=self.base_model_path,
            device=self.device,
            mixed_precision=mixed_precision,
            allow_tf32=True
        )

    def process(self, person_path, garments, output_path, steps=30, seed=42):
        print(f"Processing person: {person_path}")
        t0 = time.time()
        
        if not os.path.exists(person_path):
            raise FileNotFoundError(f"Person image not found: {person_path}")

        raw_person = Image.open(person_path).convert("RGB")
        person_img = center_crop_to_aspect_ratio(raw_person, 3/4).resize(PERSON_SIZE, Image.LANCZOS)
        
        # Run Detectors
        pose_img = self.dwpose(person_img)
        densepose_arr = np.array(self.densepose(person_img))
        lip_arr = np.array(self.schp_lip(person_img))
        atr_arr = np.array(self.schp_atr(person_img))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Validate single garment input & determine mask part
        active_garments = [k for k, v in garments.items() if v]
        if len(active_garments) != 1:
             raise ValueError(f"Please provide exactly one garment type. Found: {active_garments}")

        garment_type = active_garments[0]
        if garment_type == "upper":
            mask_part = "upper"
        elif garment_type == "lower":
            mask_part = "lower"
        elif garment_type == "dress":
            mask_part = "overall"
        else:
            # Fallback for shoe/bag to 'overall' as they are not explicitly supported 
            # keys in cloth_agnostic_mask's 'part' argument validation
            mask_part = "overall"

        # Generate Mask
        mask_img = cloth_agnostic_mask(
            densepose_arr, lip_arr, atr_arr,
            part=mask_part,
            square_cloth_mask=False
        )

        # Prepare References
        ref_images = []
        ref_labels = []
        ref_masks = []
        
        processing_order = [
            ("upper", "upper"), ("lower", "lower"), 
            ("dress", "overall"), ("shoe", "shoe"), ("bag", "bag")
        ]
        
        for key, model_label in processing_order:
            path = garments.get(key)
            target_size = ACCESSORY_SIZE if key in ["shoe", "bag"] else CLOTHING_SIZE
            
            if path and os.path.exists(path):
                img = Image.open(path).convert("RGB").resize(target_size, Image.LANCZOS)
                
                # --- Auto-Masking Reference Image ---
                # This ensures that if a full body model is provided for 'upper', 
                # we mask out the 'lower' parts so they don't leak into generation.
                
                ref_mask_arr = None
                if key == "bag":
                     # Use ATR for bag
                     atr_parse = np.array(self.schp_atr(img))
                     ref_mask_arr = part_mask_of(["Bag"], atr_parse, ATR_MAPPING)
                else:
                     # Use LIP for clothes/shoes
                     lip_parse = np.array(self.schp_lip(img))
                     
                     if model_label == "upper":
                         ref_mask_arr = part_mask_of(["Upper-clothes", "Coat", "Jumpsuits"], lip_parse, LIP_MAPPING)
                     elif model_label == "lower":
                         ref_mask_arr = part_mask_of(["Pants", "Skirt"], lip_parse, LIP_MAPPING)
                     elif model_label == "overall": # dress
                         ref_mask_arr = part_mask_of(["Dress", "Coat", "Jumpsuits", "Upper-clothes", "Skirt", "Pants"], lip_parse, LIP_MAPPING)
                     elif model_label == "shoe":
                         ref_mask_arr = part_mask_of(["Left-shoe", "Right-shoe", "Socks"], lip_parse, LIP_MAPPING)
                
                if ref_mask_arr is not None and ref_mask_arr.sum() > 0:
                     # Expand mask slightly to avoid cutting off edges
                     # ref_mask_arr = cv2.dilate(ref_mask_arr.astype(np.uint8), np.ones((5,5), np.uint8), iterations=1)
                     ref_mask_pil = Image.fromarray((ref_mask_arr * 255).astype(np.uint8)).convert("L")
                     # Apply mask to image (make background black)
                     img_bg = Image.new("RGB", img.size, (0,0,0))
                     img = Image.composite(img, img_bg, ref_mask_pil)

                ref_images.append(img)
                ref_labels.append(model_label)
                ref_masks.append(1)
            else:
                ref_images.append(Image.new("RGB", target_size, (0,0,0)))
                ref_labels.append(model_label)
                ref_masks.append(0)
        
        t1 = time.time()
        encoding_time = t1 - t0

        # Inference
        print(f"Running Inference ({steps} steps)...")
        generator = torch.Generator(device=self.device).manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            result = self.pipeline(
                person=person_img,
                mask=mask_img,
                ref_images=ref_images,
                ref_labels=ref_labels,
                ref_attention_masks=ref_masks,
                pose=pose_img,
                num_inference_steps=steps,
                guidance_scale=2.5,
                generator=generator,
                return_pil=True
            )

        t2 = time.time()
        inference_time = t2 - t1
        total_time = t2 - t0
        
        stats = {
            "encoding_time": f"{encoding_time:.2f}s",
            "inference_time": f"{inference_time:.2f}s",
            "total_time": f"{total_time:.2f}s"
        }
        
        if torch.cuda.is_available():
            max_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"Max VRAM used: {max_vram:.2f} GB")
            stats["max_vram"] = f"{max_vram:.2f} GB"

        if output_path:
            result[0].save(output_path)
            print(f"Success! Image saved to: {output_path}")
            return output_path, stats
        else:
            return result[0], stats
