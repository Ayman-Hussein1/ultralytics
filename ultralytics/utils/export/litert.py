# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER, YAML


def torch2litert(
    model: torch.nn.Module,
    im: torch.Tensor,
    file: Path | str,
    int8: bool = False,
    metadata: dict | None = None,
    prefix: str = "",
) -> Path:
    """Export a PyTorch model to LiteRT format using litert_torch, with optional INT8 quantization.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): Example input tensor for tracing.
        file (Path | str): Source model file path used to derive output directory.
        int8 (bool): Whether to apply dynamic-range INT8 quantization.
        metadata (dict | None): Optional metadata saved as ``metadata.yaml``.
        prefix (str): Prefix for log messages.

    Returns:
        (Path): Path to the exported ``_litert_model`` directory.
    """
    from ultralytics.utils.checks import check_requirements

    check_requirements(("litert-torch>=0.9.0", "ai-edge-litert>=2.1.4"))
    import litert_torch

    LOGGER.info(f"\n{prefix} starting export with litert_torch {litert_torch.__version__}...")
    file = Path(file)
    f = Path(str(file).replace(file.suffix, "_litert_model"))
    f.mkdir(parents=True, exist_ok=True)

    edge_model = litert_torch.convert(model, (im,))
    tflite_file = f / f"{file.stem}_{'int8' if int8 else 'float32'}.tflite"
    edge_model.export(tflite_file)

    if int8:
        check_requirements("ai-edge-quantizer>=0.6.0")
        from ai_edge_quantizer import quantizer, recipe

        LOGGER.info(f"{prefix} applying INT8 dynamic-range quantization...")
        qt = quantizer.Quantizer(str(tflite_file))
        qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
        qt.quantize().export_model(str(tflite_file), overwrite=True)

    YAML.save(f / "metadata.yaml", metadata or {})
    return f
