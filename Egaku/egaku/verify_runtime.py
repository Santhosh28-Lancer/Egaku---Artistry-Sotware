"""Pre-flight checks for Egaku's local AI runtime.

This file deliberately lives outside a batch-file -c command. Windows CMD
expands percent signs in inline commands, which previously corrupted the
VRAM diagnostic before Python could parse it.
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import accelerate
        import diffusers
        import huggingface_hub
        import peft
        import safetensors
        import torch
        import torchvision
        import transformers
        from diffusers.utils.import_utils import is_peft_available
    except Exception as exc:
        print(f"ERROR: A required Egaku package could not be imported: {exc}")
        print("Close Egaku and run setup_gpu.bat again.")
        return 1

    checks = [
        ("PyTorch", torch.__version__.split("+")[0], "2.13.0"),
        ("torchvision", torchvision.__version__.split("+")[0], "0.28.0"),
    ]
    for package, actual, expected in checks:
        print(f"{package}: {actual}")
        if actual != expected:
            print(f"ERROR: Expected {package} {expected}, found {actual}.")
            print("Run setup_gpu.bat to restore the pinned versions.")
            return 1

    print(f"Diffusers: {diffusers.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Accelerate: {accelerate.__version__}")
    print(f"PEFT: {peft.__version__}")
    print(f"Safetensors: {safetensors.__version__}")
    print(f"Hugging Face Hub: {huggingface_hub.__version__}")

    if not is_peft_available():
        print("ERROR: Diffusers cannot detect PEFT.")
        print("Run setup_gpu.bat, then fully restart Egaku.")
        return 1

    # CI can validate the full Python stack without claiming an NVIDIA GPU exists.
    if "--packages-only" in sys.argv:
        print("Package preflight: PASS (CUDA hardware test intentionally skipped)")
        return 0

    if torch.version.cuda is None or not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable to PyTorch.")
        print("Check the NVIDIA driver, then run setup_gpu.bat again.")
        return 1

    try:
        device = torch.device("cuda:0")
        probe = torch.ones((256, 256), device=device, dtype=torch.float16)
        _ = probe @ probe
        torch.cuda.synchronize(device)
        del probe
        torch.cuda.empty_cache()
        properties = torch.cuda.get_device_properties(device)
    except Exception as exc:
        print(f"ERROR: NVIDIA CUDA compute test failed: {exc}")
        print("Update/check the NVIDIA driver and restart Windows.")
        return 1

    memory_gb = properties.total_memory / (1024 ** 3)
    print(f"GPU: {properties.name}")
    print(f"VRAM: {memory_gb:.2f} GB")
    print(f"CUDA runtime: {torch.version.cuda}")
    print("PEFT backend: available")
    print("CUDA compute test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
