# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
from pathlib import Path

import numpy as np
from ultralytics.utils import LOGGER


class AFSSScheduler:
    """Anti-Forgetting Sampling Strategy (AFSS) scheduler for YOLO training.

    This scheduler partitions images into easy, moderate, and hard sets based on
    per-image precision and recall, then samples from each set according to a
    budget policy that prevents forgetting by periodically forcing long-unseen
    images back into the training batch.

    Attributes:
        num_images (int): Total number of images in the dataset.
        tau (int): Warmup epoch count (ceiling of warmup_epochs).
        seed (int): Random seed for deterministic sampling.
        state (dict[int, dict]): Per-image state containing precision, recall, and last_seen_epoch.
    """

    def __init__(self, num_images: int, warmup_epochs: float = 3.0, seed: int = 0):
        """Initialize AFSSScheduler with the given dataset size and warmup configuration.

        Args:
            num_images (int): Total number of images in the dataset.
            warmup_epochs (float): Number of warmup epochs before AFSS sampling activates.
            seed (int): Random seed for deterministic sampling.
        """
        self.num_images = num_images
        self.tau = math.ceil(warmup_epochs)
        self.seed = seed
        self.state = {i: {"precision": 0.0, "recall": 0.0, "last_seen_epoch": -1} for i in range(num_images)}

    def sample_indices(self, epoch: int) -> list[int]:
        """Sample image indices for the given epoch according to AFSS policy.

        Args:
            epoch (int): Current training epoch.

        Returns:
            (list[int]): List of selected image indices.
        """
        rng = np.random.RandomState(epoch + self.seed)
        selected = []

        easy_set = []
        moderate_set = []
        hard_set = []

        for i, st in self.state.items():
            s_i = min(st["precision"], st["recall"])
            if s_i > 0.85:
                easy_set.append(i)
            elif s_i >= 0.55:
                moderate_set.append(i)
            else:
                hard_set.append(i)

        # Hard set: include all hard images
        selected.extend(hard_set)

        # Easy set
        if easy_set:
            forced_easy = [
                i for i in easy_set if (epoch - 1 - self.state[i]["last_seen_epoch"]) >= 10
            ]  # spec-defined formula
            easy_budget = round(0.02 * len(easy_set))
            forced_easy_quota = min(len(forced_easy), math.floor(0.5 * easy_budget))
            random_easy_quota = max(easy_budget - forced_easy_quota, 0)

            if easy_budget > 0:
                forced_easy_sample = []
                if forced_easy_quota > 0 and forced_easy:
                    forced_easy_sample = rng.choice(forced_easy, size=forced_easy_quota, replace=False).tolist()
                selected.extend(forced_easy_sample)

                remaining_easy = [i for i in easy_set if i not in forced_easy_sample]
                if random_easy_quota > 0 and remaining_easy:
                    random_easy_sample = rng.choice(
                        remaining_easy, size=min(random_easy_quota, len(remaining_easy)), replace=False
                    ).tolist()
                    selected.extend(random_easy_sample)

        # Moderate set
        if moderate_set:
            forced_moderate = [
                i
                for i in moderate_set
                if (epoch - 1 - self.state[i]["last_seen_epoch"]) >= 3  # spec-defined formula
            ]
            M1 = round(0.4 * len(moderate_set)) - len(forced_moderate)
            random_moderate_quota = max(min(len(moderate_set) - len(forced_moderate), M1), 0)

            selected.extend(forced_moderate)

            remaining_moderate = [i for i in moderate_set if i not in forced_moderate]
            if random_moderate_quota > 0 and remaining_moderate:
                random_moderate_sample = rng.choice(
                    remaining_moderate, size=min(random_moderate_quota, len(remaining_moderate)), replace=False
                ).tolist()
                selected.extend(random_moderate_sample)

        selected_indices = sorted(selected)
        if not selected_indices:
            LOGGER.warning(f"AFSS sampled zero images for epoch {epoch}; falling back to full dataset.")
            selected_indices = list(range(self.num_images))
        return selected_indices

    def update_last_seen(self, indices: list[int], epoch: int) -> None:
        """Update last_seen_epoch for the given indices.

        Args:
            indices (list[int]): List of image indices that were seen this epoch.
            epoch (int): Current training epoch.
        """
        for i in indices:
            self.state[i]["last_seen_epoch"] = epoch

    def update_metrics(self, image_metrics: dict[str, dict], filename_to_idx: dict[str, int]) -> None:
        """Update per-image precision and recall from validator metrics.

        Args:
            image_metrics (dict[str, dict]): Dict keyed by image filename with precision/recall values.
            filename_to_idx (dict[str, int]): Mapping from filename to dataset index.
        """
        for filename, metrics in image_metrics.items():
            idx = filename_to_idx.get(Path(filename).name)
            if idx is None:
                continue
            self.state[idx]["precision"] = float(metrics.get("precision", 0.0))
            self.state[idx]["recall"] = float(metrics.get("recall", 0.0))
