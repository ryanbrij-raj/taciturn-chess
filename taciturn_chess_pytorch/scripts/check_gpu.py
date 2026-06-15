"""
scripts/check_gpu.py — Verify PyTorch can see your GPU.
Run with: python scripts/check_gpu.py
"""
import sys

def check_gpu():
    print("=" * 50)
    print("  Taciturn Chess — GPU Check (PyTorch)")
    print("=" * 50)

    print(f"\n✓ Python {sys.version}")

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed. Run:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"\n✓ Found {count} GPU(s):")
        for i in range(count):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   [{i}] {name} ({mem:.1f} GB VRAM)")

        print("\n  Running quick GPU compute test...")
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        c = torch.mm(a, b)
        print(f"✓ Matrix multiply (1000×1000) on GPU: OK — result shape {list(c.shape)}")
        print("\n✓ All checks passed. You're ready to train!")
        print("  Run: python train.py")
    else:
        print("\n✗ No GPU detected.")
        print("  Make sure you installed the CUDA version of PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

    print("=" * 50)

if __name__ == "__main__":
    check_gpu()
