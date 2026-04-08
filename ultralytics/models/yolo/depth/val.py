# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth estimation validator for YOLO models."""

from __future__ import annotations

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER


class DepthValidator(DetectionValidator):
    """Validator for YOLO depth estimation models.

    Overrides detection-specific validation (NMS, mAP) with depth metrics
    (delta1, abs_rel, rmse, silog).
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize DepthValidator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "depth"
        self.metrics_list = []  # per-image metrics

    def postprocess(self, preds):
        """No NMS needed for depth — return predictions as-is."""
        return preds

    def init_metrics(self, model):
        """Initialize depth-specific metrics."""
        self.metrics_list = []

    def update_metrics(self, preds, batch):
        """Compute per-batch depth metrics (no-op during training validation)."""
        pass  # Depth metrics computed in get_stats

    def get_stats(self):
        """Return empty stats (depth uses loss as the primary metric)."""
        return {}

    def print_results(self):
        """Print depth validation summary."""
        pass

    def get_desc(self):
        """Return description for progress bar."""
        return ""
