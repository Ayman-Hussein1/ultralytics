#!/usr/bin/env python
"""Trace teacher models to TorchScript .ts files for zero-dependency training.

Follows the MobileCLIP pattern (ultralytics/nn/image_model.py: MobileCLIPImageTS loads .ts via
torch.jit.load). Each teacher is wrapped to return a (cls, patches) tuple suitable for tracing.

Usage:
    python export_teachers.py                    # trace all available teachers
    python export_teachers.py eupe:vitb16        # trace a single teacher
    python export_teachers.py --output-dir ./ts  # custom output directory

Output .ts files can be loaded via TorchScriptTeacher in teacher_model.py.
"""

import argparse
import os
from pathlib import Path

# Load HF token from .env before any HF imports (needed for gated models like DINOv3)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from ultralytics.nn.teacher_model import TEACHER_REGISTRY, build_teacher_model  # noqa: E402


class _TraceWrapper(nn.Module):
    """Wrap a teacher to return (cls, patches) tuple for TorchScript tracing.

    TorchScript tracing works best with simple tuple outputs. The Python TeacherModel returns a TeacherOutput dataclass;
    this wrapper converts to a plain tuple for tracing. For patches-only teachers (SAM3, ConvNeXt), cls is a zero tensor
    (same device/dtype as patches).
    """

    def __init__(self, teacher):
        """Initialize wrapper around a native teacher model.

        Args:
            teacher (TeacherModel): Native Python teacher model to wrap.
        """
        super().__init__()
        self.teacher = teacher
        self.token_types = teacher.token_types

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run teacher and return (cls, patches) tuple.

        Args:
            x (torch.Tensor): Preprocessed image tensor (B, 3, H, W).

        Returns:
            (tuple): (cls (B, D), patches (B, N, D)). cls is zeros if patches-only teacher.
        """
        out = self.teacher.model.forward_features(x) if hasattr(self.teacher.model, "forward_features") else None

        if out is not None:
            patches = out["x_norm_patchtokens"]
            cls = (
                out["x_norm_clstoken"]
                if "cls" in self.token_types
                else torch.zeros(patches.shape[0], patches.shape[2], device=patches.device, dtype=patches.dtype)
            )
        else:
            # Fallback for non-EUPE teachers (DINOv3 via transformers, SigLIP2, SAM3)
            teacher_out = self.teacher.encode(x)
            patches = teacher_out.patches
            cls = (
                teacher_out.cls
                if teacher_out.cls is not None
                else torch.zeros(patches.shape[0], patches.shape[2], device=patches.device, dtype=patches.dtype)
            )
        return cls, patches


def trace_teacher(variant: str, output_dir: Path, device: str = "cpu"):
    """Trace a single teacher model to TorchScript.

    Args:
        variant (str): Teacher variant string (e.g. 'eupe:vitb16').
        output_dir (Path): Directory to save the .ts file.
        device (str): Device for tracing ('cpu' for portability).
    """
    reg = TEACHER_REGISTRY[variant]
    ts_name = variant.replace(":", "_") + ".ts"
    ts_path = output_dir / ts_name

    print(f"Tracing {variant} ({reg['embed_dim']}d, {reg['num_patches']} patches)...")
    teacher = build_teacher_model(variant, torch.device(device))
    wrapper = _TraceWrapper(teacher)
    wrapper.eval()

    imgsz = reg["imgsz"]
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)

    # Verify
    cls_out, patch_out = traced(dummy)
    print(f"  CLS: {cls_out.shape}, Patches: {patch_out.shape}")

    torch.jit.save(traced, str(ts_path))
    size_mb = ts_path.stat().st_size / 1e6
    print(f"  Saved: {ts_path} ({size_mb:.1f} MB)")
    return ts_path


def main():
    """Trace teacher models to TorchScript."""
    parser = argparse.ArgumentParser(description="Trace teacher models to TorchScript .ts files")
    parser.add_argument("variants", nargs="*", help="Teacher variants to trace (default: all available)")
    parser.add_argument("--output-dir", type=Path, default=Path("teacher_weights"), help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device for tracing")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = args.variants or list(TEACHER_REGISTRY.keys())

    for v in variants:
        if v not in TEACHER_REGISTRY:
            print(f"Unknown variant '{v}', skipping. Available: {list(TEACHER_REGISTRY.keys())}")
            continue
        try:
            trace_teacher(v, args.output_dir, args.device)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    print(f"\nDone. .ts files saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
