# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Shared ReID encoder used by BoT-SORT, Deep OC-SORT, and TrackTrack.

Wraps `AutoBackend` so the model can be a TorchScript / ONNX / TensorRT / OpenVINO export and
the same code path serves all of them. Output is a normalized embedding per detection crop.
"""

from __future__ import annotations

import numpy as np
import torch

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import save_one_box


class ReID:
    """ReID encoder backed by `AutoBackend` (TorchScript / ONNX / TensorRT / etc.)."""

    def __init__(self, model: str, imgsz: int = 224, device: str | torch.device | None = None, fp16: bool = False):
        """Initialize encoder for re-identification.

        Args:
            model (str): Path to an exported ReID model that outputs an embedding tensor.
            imgsz (int): Square input size used for crop preprocessing.
            device (str | torch.device | None): Inference device; defaults to CUDA if available.
            fp16 (bool): Use half precision when the backend supports it.
        """
        self.imgsz = imgsz
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoBackend(str(model), device=self.device, fp16=fp16, verbose=False)
        self.fp16 = self.model.fp16

    def _crops_to_tensor(self, img: np.ndarray, dets: np.ndarray) -> torch.Tensor:
        """Crop detections from img and stack into a normalized BCHW float tensor at self.imgsz."""
        crops = [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]
        size = self.imgsz
        batch = torch.empty(len(crops), 3, size, size, dtype=torch.float32)
        for i, c in enumerate(crops):
            t = torch.from_numpy(np.ascontiguousarray(c[..., ::-1])).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[i] = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0]
        batch = batch.to(self.device)
        return batch.half() if self.fp16 else batch

    @torch.inference_mode()
    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        """Extract embeddings for detected objects."""
        feats = self.model(self._crops_to_tensor(img, dets))
        return [f.cpu().numpy() for f in feats]
