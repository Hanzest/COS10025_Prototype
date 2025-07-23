# Test CUDA availability and compatibility
# Print GPU driver version and device information

import torch
import os

# Print driver version:
driver_version = os.popen('nvidia-smi --query-gpu=driver_version --format=csv,noheader').read().strip()
print(f"GPU driver version: {driver_version}")

# Check driver compatibility
def check_driver_compatibility(driver_ver):
    try:
        major_ver = int(driver_ver.split('.')[0])
        if major_ver >= 520:
            return "✅ Excellent - Supports CUDA 12.0+ and latest PyTorch"
        elif major_ver >= 470:
            return "✅ Good - Supports CUDA 11.4+ and PyTorch 1.12+"
        elif major_ver >= 450:
            return "⚠️  Moderate - Supports CUDA 11.0+ and PyTorch 1.8+"
        else:
            return "❌ Too old - Only supports CUDA 10.x, incompatible with modern PyTorch/YOLOv8"
    except:
        return "❓ Unable to parse version"

print(f"Driver Status: {check_driver_compatibility(driver_version)}")

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("\n=== SOLUTION ===")
    print("To use GPU acceleration with YOLOv8:")
    print("1. Update NVIDIA driver to version 470+ (recommended: latest)")
    print("2. Download from: https://www.nvidia.com/drivers")
    print("3. After update, restart computer")
    print("4. Current PyTorch will then work with GPU")
    print("\nAlternatively: Use CPU training (slower but functional)")