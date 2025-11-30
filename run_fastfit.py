import argparse
import os
import sys
import torch
from PIL import Image
import numpy as np


# Add current directory to sys.path
sys.path.append(os.getcwd())

try:
    from models.fastfit_pipeline import FastFitPipeline
    from models.parse_utils import DWposeDetector, DensePose, SCHP
    from models.parse_utils.automasker import multi_ref_cloth_agnostic_mask, cloth_agnostic_mask
    from utils import resize_and_crop, resize_and_padding
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you are running this script from the root of the FastFit repository.")
    sys.exit(1)

def load_models(args):
    print(f"Loading FastFit model from {args.fastfit_model_path}...")
    fastfit_path = args.fastfit_model_path

    pipeline = FastFitPipeline(
        base_model_path=fastfit_path,
        device=args.device,
        mixed_precision="bf16" if args.device == "cuda" else "fp32"
    )

    print(f"Loading Human Toolkit from {args.human_toolkit_path}...")
    human_toolkit_path = args.human_toolkit_path

    dwpose = DWposeDetector(
        pretrained_model_name_or_path=os.path.join(human_toolkit_path, "DWPose"),
        device="cpu" # Often simpler to keep on CPU
    )
    densepose = DensePose(
        model_path=os.path.join(human_toolkit_path, "DensePose"), 
        device=args.device
    )
    schp_lip = SCHP(
        ckpt_path=os.path.join(human_toolkit_path, "SCHP", "schp-lip.pth"),
        device=args.device
    )
    schp_atr = SCHP(
        ckpt_path=os.path.join(human_toolkit_path, "SCHP", "schp-atr.pth"),
        device=args.device
    )
    
    return pipeline, dwpose, densepose, schp_lip, schp_atr

def main():
    parser = argparse.ArgumentParser(description="Run FastFit Virtual Try-On Pipeline")
    parser.add_argument("--person_image", type=str, required=True, help="Path to the person image")
    parser.add_argument("--output_path", type=str, default="output.png", help="Path to save the output image")
    
    # Garment Inputs
    parser.add_argument("--garment_upper", type=str, help="Path to upper garment image")
    parser.add_argument("--garment_lower", type=str, help="Path to lower garment image")
    parser.add_argument("--garment_dress", type=str, help="Path to dress/overall garment image")
    parser.add_argument("--garment_shoe", type=str, help="Path to shoe image")
    parser.add_argument("--garment_bag", type=str, help="Path to bag image")

    # Masking Controls
    parser.add_argument("--mask_type", type=str, default="contour", choices=["contour", "box"], help="Mask shape: 'contour' (strict) or 'box' (loose/rectangular)")
    parser.add_argument("--mask_expand", action="store_true", help="Enable horizon expansion for box masks (useful for oversized clothes)")
    parser.add_argument("--mask_mode", type=str, default="auto", choices=["auto", "upper", "lower", "overall", "inner", "outer"], help="Masking mode/part to protect. 'auto' uses multi-ref mode.")

    # Model/System Config
    parser.add_argument("--fastfit_model_path", type=str, default="Models/FastFit-MR-1024", help="Path or Repo ID for FastFit model")
    parser.add_argument("--human_toolkit_path", type=str, default="Models/Human-Toolkit", help="Path or Repo ID for Human Toolkit")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")

    args = parser.parse_args()

    # Validate that at least one garment is provided
    if not any([args.garment_upper, args.garment_lower, args.garment_dress, args.garment_shoe, args.garment_bag]):
        print("Error: You must provide at least one garment image (e.g. --garment_upper)")
        sys.exit(1)

    # Load Models
    pipeline, dwpose, densepose, schp_lip, schp_atr = load_models(args)

    # Load Person Image
    print("Processing person image...")
    person_pil = Image.open(args.person_image).convert("RGB")
    person_pil = resize_and_crop(person_pil, (768, 1024))
    
    # Process Garments
    ref_images = []
    ref_labels = []
    ref_attention_masks = []
    
    # Mapping of argument names to internal labels and image processing logic
    garment_inputs = [
        ("upper", args.garment_upper),
        ("lower", args.garment_lower),
        ("overall", args.garment_dress),
        ("shoe", args.garment_shoe),
        ("bag", args.garment_bag)
    ]

    for label, path in garment_inputs:
        if path:
            print(f"Processing {label} garment: {path}")
            g_pil = Image.open(path).convert("RGB")
            # Resize logic: shoes and bags get smaller target size
            target_size = (384, 512) if label in ["shoe", "bag"] else (768, 1024)
            g_pil = resize_and_padding(g_pil, target_size)
            
            ref_images.append(g_pil)
            ref_labels.append(label)
            ref_attention_masks.append(1)

    # Run Parsers
    print("Running Human Parsers...")
    pose_pil = dwpose(person_pil)
    densepose_pil = densepose(person_pil)
    lip_pil = schp_lip(person_pil)
    atr_pil = schp_atr(person_pil)

    # Prepare for Automasker
    # DensePose: RGB IUV -> use I (channel 0) as index
    densepose_arr = np.array(densepose_pil)
    if densepose_arr.ndim == 3:
        densepose_arr = densepose_arr[:, :, 0]
    
    # SCHP: P mode -> indices
    lip_arr = np.array(lip_pil)
    if lip_arr.ndim == 3:
        lip_arr = lip_arr[:, :, 0]
        
    atr_arr = np.array(atr_pil)
    if atr_arr.ndim == 3:
        atr_arr = atr_arr[:, :, 0]

    # Generate Mask
    print(f"Generating Mask (Mode: {args.mask_mode}, Type: {args.mask_type})...")
    square_mask = (args.mask_type == "box")
    
    if args.mask_mode == "auto":
        mask_pil = multi_ref_cloth_agnostic_mask(
            densepose_arr,
            lip_arr,
            atr_arr,
            square_cloth_mask=square_mask,
            horizon_expand=args.mask_expand
        )
    else:
        mask_pil = cloth_agnostic_mask(
            densepose_arr,
            lip_arr,
            atr_arr,
            part=args.mask_mode,
            square_cloth_mask=square_mask
        )

    # Run FastFit
    print("Running FastFit Inference...")
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    
    result = pipeline(
        person=person_pil,
        mask=mask_pil,
        ref_images=ref_images,
        ref_labels=ref_labels,
        ref_attention_masks=ref_attention_masks,
        pose=pose_pil,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        return_pil=True
    )[0]

    result.save(args.output_path)
    print(f"Successfully saved result to {args.output_path}")

if __name__ == "__main__":
    main()
