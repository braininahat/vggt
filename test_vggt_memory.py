#!/usr/bin/env python3
"""
Test VGGT model memory usage and find appropriate batch size.
"""

import torch
import gc
from vggt.models.vggt import VGGT

def test_model_memory(model_name="facebook/VGGT-S", batch_size=1, seq_length=8):
    """Test model with different configurations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\nTesting {model_name} with batch_size={batch_size}, seq_length={seq_length}")
    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    try:
        # Load model
        print("Loading model...")
        model = VGGT.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        # Create dummy input
        dummy_images = torch.randn(batch_size, seq_length, 3, 518, 518).to(device)
        print(f"Input created. GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        # Forward pass
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(dummy_images)
        
        print(f"Forward pass complete. GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print("Output keys:", outputs.keys())
        
        # Print output shapes
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM Error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if 'dummy_images' in locals():
            del dummy_images
        if 'outputs' in locals():
            del outputs
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    
    # Test different configurations
    configs = [
        ("facebook/VGGT-S", 1, 8),  # Small model, batch 1
        ("facebook/VGGT-S", 2, 8),  # Small model, batch 2
        ("facebook/VGGT-S", 1, 4),  # Small model, fewer frames
        ("facebook/VGGT-B", 1, 8),  # Base model
        ("facebook/VGGT-B", 1, 4),  # Base model, fewer frames
    ]
    
    for model_name, batch_size, seq_length in configs:
        success = test_model_memory(model_name, batch_size, seq_length)
        if not success:
            print(f"Failed with {model_name}, batch_size={batch_size}, seq_length={seq_length}")
            break
        print("-" * 80)


if __name__ == "__main__":
    main()