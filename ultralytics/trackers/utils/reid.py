# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Shared ReID encoder used by BoT-SORT, Deep OC-SORT, and TrackTrack.

Accepts a TorchScript embedding model (preferred) or a YOLO .pt checkpoint
(legacy YOLO-predictor path). Returns one normalized embedding per detection.
"""

from __future__ import annotations

import numpy as np
import torch

from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import save_one_box


class ReID:
    """ReID encoder. Loads a TorchScript embedding model, or falls back to a YOLO checkpoint."""

    def __init__(self, model: str, imgsz: int = 224, device: str | torch.device | None = None):
        """Initialize encoder for re-identification.

        Args:
            model (str): Path to a TorchScript (.torchscript) reid model, or a YOLO .pt checkpoint.
            imgsz (int): Square input size used by the TorchScript model.
            device (str | torch.device | None): Device for inference; defaults to CUDA if available.
        """
        self.imgsz = imgsz
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_torchscript = str(model).endswith(".torchscript")

        if self.is_torchscript:
            self.model = torch.jit.load(str(model), map_location=self.device).eval()
        else:
            from ultralytics import YOLO

            self.model = YOLO(model)
            self.model(embed=[len(self.model.model.model) - 2 if ".pt" in model else -1], verbose=False, save=False)

    def _crops_to_tensor(self, img: np.ndarray, dets: np.ndarray) -> torch.Tensor:
        """Crop detections from img and stack into a normalized BCHW float tensor at self.imgsz."""
        crops = [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]
        size = self.imgsz
        batch = torch.empty(len(crops), 3, size, size, dtype=torch.float32)
        for i, c in enumerate(crops):
            t = torch.from_numpy(np.ascontiguousarray(c[..., ::-1])).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            batch[i] = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0]
        return batch.to(self.device)

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        """Extract embeddings for detected objects."""
        if self.is_torchscript:
            with torch.inference_mode():
                feats = self.model(self._crops_to_tensor(img, dets))
            return [f.cpu().numpy() for f in feats]

        feats = self.model.predictor(
            [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]
        )
        if len(feats) != dets.shape[0] and feats[0].shape[0] == dets.shape[0]:
            feats = feats[0]  # batched prediction with non-PyTorch backend
        return [f.cpu().numpy() for f in feats]
