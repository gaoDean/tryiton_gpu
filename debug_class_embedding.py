import torch
import sys
import os

# Ensure we can import from the module folder
sys.path.append(os.getcwd())

from module.unet_2d_condition import UNet2DConditionModel

def debug_class_embeddings():
    print("--- Starting Class Embedding Debugger ---")
    
    # 1. Initialize a dummy UNet with Class Embeddings enabled
    # We don't need to load weights; we just need the architecture logic.
    print("1. Initializing UNet with num_class_embeds=5...")
    try:
        unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=4,
            out_channels=4,
            # Vital params for Class Embedding:
            num_class_embeds=5,             # 5 classes (upper, lower, etc.)
            class_embed_type=None,          # Default behavior for basic embedding
            time_embedding_type="positional",
            block_out_channels=(32, 64),    # Small size for speed
            layers_per_block=1,
            cross_attention_dim=32,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D") # Adjust to match block_out_channels length
        )
        print("   [OK] UNet initialized.")
    except Exception as e:
        print(f"   [FAIL] Could not initialize UNet: {e}")
        return

    # 2. Inspect the Embedding Layer
    print("\n2. Inspecting Internal Structure...")
    if hasattr(unet, 'class_embedding') and isinstance(unet.class_embedding, torch.nn.Embedding):
        print(f"   [OK] Found 'class_embedding' layer: {unet.class_embedding}")
    else:
        print(f"   [FAIL] 'class_embedding' layer missing or wrong type. Got: {getattr(unet, 'class_embedding', 'None')}")
        return

    # 3. Test Embedding Generation (The Ablation Logic)
    print("\n3. Testing Vector Generation (Ablation Logic)...")
    
    # Create dummy sample (needed for dtype/device matching in the function)
    dummy_sample = torch.randn(1, 4, 64, 64)
    
    # Label 0: Upper
    label_upper = torch.tensor([0])
    emb_upper = unet.get_class_embed(dummy_sample, label_upper)
    
    # Label 1: Lower
    label_lower = torch.tensor([1])
    emb_lower = unet.get_class_embed(dummy_sample, label_lower)
    
    print(f"   Vector for Label 0 (Upper) shape: {emb_upper.shape}")
    print(f"   Vector for Label 1 (Lower) shape: {emb_lower.shape}")

    # 4. Verification: Are they different?
    print("\n4. Verifying Differentiation...")
    if torch.allclose(emb_upper, emb_lower):
        print("   [FAIL] The embeddings for Upper (0) and Lower (1) are IDENTICAL!")
        print("   This means the mechanism is NOT working. The model cannot distinguish garments.")
    else:
        diff = (emb_upper - emb_lower).abs().mean().item()
        print(f"   [OK] Embeddings are distinct.")
        print(f"   Average difference between vectors: {diff:.6f}")
        print("   (This proves the control mechanism from Figure 10 is active)")

    # 5. Simulate the Signal Injection
    print("\n5. Simulating Logic Injection...")
    # In the code, if class_labels is present, it should replace/mix with time embeddings.
    # We manually check the logic found in unet_2d_condition.py
    
    # Standard Time Embedding
    t_step = torch.tensor([50])
    t_emb = unet.time_embedding(unet.get_time_embed(dummy_sample, t_step), None)
    
    print(f"   Standard Time Emb Shape:  {t_emb.shape}")
    print(f"   Reference Class Emb Shape: {emb_upper.shape}")
    
    if t_emb.shape == emb_upper.shape:
        print("   [OK] Shapes match. The Class Embedding is compatible to replace Time Embedding.")
    else:
        print("   [FAIL] Shapes mismatch! The UNet will crash if it tries to swap them.")

    print("\n--- Debugging Complete ---")

if __name__ == "__main__":
    debug_class_embeddings()
