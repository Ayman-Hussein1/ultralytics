# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""ImageNet kNN evaluation for measuring backbone feature quality.

Standard protocol used by DINO, DINOv2, EUPE, and AM-RADIO for evaluating self-supervised and
distilled vision encoders. Extract L2-normalized CLS features from a frozen backbone, then
classify via temperature-weighted k-nearest-neighbor voting on ImageNet-1k.

Algorithm (Wu et al., "Unsupervised Feature Learning via Non-Parametric Instance Discrimination",
arXiv:1805.01978, Section 3.4 -- also adopted by DINO arXiv:2104.14294 and all successors):
  1. L2-normalize all features (train + val)
  2. For each val sample, find k nearest train neighbors by dot-product similarity
  3. Weight each neighbor's vote by exp(similarity / temperature)
  4. Predicted class = argmax of accumulated weighted votes

Implementation follows RADIO's single-GPU kNN (NVlabs/RADIO examples/knn_classification.py:193-336)
using the efficient scatter_add_ voting pattern from _get_vote_cls (line 193-209).
EUPE's distributed KnnModule (facebookresearch/eupe eupe/eval/knn.py:96-184) uses mathematically
equivalent softmax(sim/T) voting but with multi-GPU broadcast/gather -- unnecessary for our setup.

Default hyperparameters: k=20, T=0.07 (used by DINO, EUPE Table 1, RADIO, DINOv2).
"""

import logging

import torch
import torch.nn.functional as F

from ultralytics.data import ClassificationDataset
from ultralytics.data.build import build_dataloader

logger = logging.getLogger("ultralytics")


@torch.no_grad()
def extract_features(model, dataloader, device):
    """Extract L2-normalized CLS features from backbone.

    Feature path matches ImageEncoderModel.loss() (nn/image_encoder.py:201-208): backbone layers 0-9 -> Classify head
    conv (512->1280, 1x1) -> AdaptiveAvgPool2d(1) -> flatten. L2-normalization follows RADIO _build_database
    (examples/knn_classification.py:355) and EUPE ModelWithNormalize (eupe/eval/utils.py:30-36).

    Uses fp32 for feature extraction (matching our val precision rules in val_image_encoder.py:61-72 and UNIC convention
    at unic/main_unic.py:432).

    Args:
        model: YOLO classification model (ClassificationModel or ImageEncoderModel).
        dataloader: DataLoader yielding {"img": tensor, "cls": tensor}.
        device: torch.device for computation.

    Returns:
        (tuple): (features, labels) tensors on CPU. features shape (N, 1280), labels shape (N,).
    """
    was_training = model.training
    model.eval()
    all_features, all_labels = [], []

    for batch in dataloader:
        imgs = batch["img"].to(device, non_blocking=True).float()
        labels = batch["cls"].to(device, non_blocking=True)

        # Backbone forward: layers 0-9 (Conv, C3k2, C2PSA etc.), same path as image_encoder.py:201-203
        x = imgs
        for m in model.model[:-1]:
            x = m(x)

        # CLS feature via Classify head internals (image_encoder.py:205-208)
        # head.conv: Conv(512->1280, k=1) on 7x7 feature map
        # head.pool: AdaptiveAvgPool2d(1) -> global average = CLS-equivalent token
        head = model.model[-1]
        features = head.pool(head.conv(x)).flatten(1)  # (B, 1280)

        # L2-normalize per RADIO _build_database:355 and EUPE ModelWithNormalize
        features = F.normalize(features, dim=1, p=2)

        all_features.append(features.cpu())
        all_labels.append(labels.cpu().long().squeeze())

    if was_training:
        model.train()
    return torch.cat(all_features), torch.cat(all_labels)


def knn_accuracy(
    train_features,
    train_labels,
    val_features,
    val_labels,
    k=20,
    temp=0.07,
    num_classes=1000,
    chunk_size=256,
    device=None,
):
    """Compute kNN top-1 accuracy using temperature-weighted voting.

    For each val image, find k nearest neighbors in train set by cosine similarity (dot product of L2-normalized
    features), weight by exp(sim / temp), accumulate class votes via scatter_add_, predict via argmax.

    Voting follows RADIO _get_vote_cls (examples/knn_classification.py:193-209): weights = exp(sim / 0.07)
    cls_vec.scatter_add_(dim=1, index=labels, src=weights) vote_id = argmax(cls_vec) This is mathematically equivalent
    to EUPE's softmax(sim/T) voting (eupe/eval/knn.py:178) since the softmax denominator is constant across classes and
    cancels in argmax.

    Chunked computation avoids materializing full (N_val x N_train) similarity matrix. With chunk_size=256 and
    N_train=1.28M: ~1.3 GB GPU memory per chunk.

    Args:
        train_features (Tensor): L2-normalized train features (N_train, D) on CPU.
        train_labels (Tensor): Train labels (N_train,) on CPU.
        val_features (Tensor): L2-normalized val features (N_val, D) on CPU.
        val_labels (Tensor): Val labels (N_val,) on CPU.
        k (int): Number of nearest neighbors.
        temp (float): Temperature for softmax weighting.
        num_classes (int): Number of classes.
        chunk_size (int): Val images processed per GPU batch.
        device: torch.device for computation. Defaults to cuda if available.

    Returns:
        (float): Top-1 accuracy as percentage (0-100).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_features_gpu = train_features.to(device)  # ~6.5 GB for 1.28M x 1280 x fp32
    train_labels_gpu = train_labels.to(device)
    correct = 0
    total = 0

    for i in range(0, len(val_features), chunk_size):
        chunk_feats = val_features[i : i + chunk_size].to(device)
        chunk_labels = val_labels[i : i + chunk_size].to(device)

        # Cosine similarity via dot product (features are L2-normalized)
        sims = chunk_feats @ train_features_gpu.T  # (chunk, N_train)
        topk_sims, topk_idx = sims.topk(k, dim=1)  # (chunk, k)
        topk_labels = train_labels_gpu[topk_idx]  # (chunk, k)

        # Temperature-weighted voting per RADIO _get_vote_cls:199-208
        weights = torch.exp(topk_sims / temp)
        cls_votes = torch.zeros(chunk_feats.shape[0], num_classes, dtype=weights.dtype, device=device)
        cls_votes.scatter_add_(dim=1, index=topk_labels, src=weights)
        preds = cls_votes.argmax(dim=1)

        correct += (preds == chunk_labels).sum().item()
        total += chunk_labels.shape[0]

    return 100.0 * correct / total


def knn_callback(data_path, k=20, temp=0.07, every_n_epochs=5, batch_size=256, workers=8):
    """Create a callback that evaluates kNN accuracy on ImageNet after each N epochs.

    Intended for on_fit_epoch_end event during Phase 1 distillation. Logs knn/top1 to both trainer.metrics (for CSV via
    save_metrics) and WandB (via wandb.log without commit, bundled with the next commit=True from wb.py:155
    on_fit_epoch_end handler).

    Args:
        data_path (str): Path to ImageNet root (must contain train/ and val/ subdirs).
        k (int): Number of nearest neighbors.
        temp (float): Temperature for softmax weighting.
        every_n_epochs (int): Run kNN eval every N epochs. ~18 min per eval.
        batch_size (int): Batch size for feature extraction.
        workers (int): DataLoader workers.

    Returns:
        (callable): Callback function for on_fit_epoch_end event.

    Examples:
        >>> from ultralytics.utils.knn_eval import knn_callback
        >>> model.add_callback("on_fit_epoch_end", knn_callback("/data/shared-datasets/imagenet"))
    """
    from pathlib import Path
    from types import SimpleNamespace

    state = {}  # mutable closure state for caching dataloaders across epochs

    def _callback(trainer):
        epoch = trainer.epoch + 1  # 1-indexed
        is_final = epoch >= trainer.epochs
        if not is_final and epoch % every_n_epochs != 0:
            return

        # Build dataloaders on first call, cache for subsequent epochs
        if "train_loader" not in state:
            root = Path(data_path)
            args = SimpleNamespace(
                imgsz=224,
                cache=False,
                fraction=1.0,
                auto_augment="",
                erasing=0.0,
                crop_fraction=1.0,
                scale=0.92,
                fliplr=0.5,
                flipud=0.0,
                hsv_h=0.015,
                hsv_s=0.4,
                hsv_v=0.4,
            )
            train_ds = ClassificationDataset(str(root / "train"), args=args, augment=False, prefix="knn-train")
            val_ds = ClassificationDataset(str(root / "val"), args=args, augment=False, prefix="knn-val")
            state["train_loader"] = build_dataloader(train_ds, batch_size, workers, shuffle=False, rank=-1)
            state["val_loader"] = build_dataloader(val_ds, batch_size, workers, shuffle=False, rank=-1)
            state["num_classes"] = len(train_ds.base.classes)
            logger.info(f"kNN eval: {len(train_ds)} train, {len(val_ds)} val, {state['num_classes']} classes")

        model = trainer.ema.ema if hasattr(trainer, "ema") and trainer.ema else trainer.model
        device = trainer.device

        logger.info(f"kNN eval: extracting train features (epoch {epoch})...")
        train_feats, train_labels = extract_features(model, state["train_loader"], device)
        logger.info("kNN eval: extracting val features...")
        val_feats, val_labels = extract_features(model, state["val_loader"], device)

        top1 = knn_accuracy(
            train_feats,
            train_labels,
            val_feats,
            val_labels,
            k=k,
            temp=temp,
            num_classes=state["num_classes"],
            device=device,
        )
        logger.info(f"kNN eval: top-1 = {top1:.2f}% (epoch {epoch}, k={k})")

        # Log to trainer.metrics (picked up by save_metrics for CSV)
        trainer.metrics["knn/top1"] = round(top1, 4)

        # Log to WandB directly without commit -- gets bundled with the next commit=True
        # from the WandB on_fit_epoch_end handler (utils/callbacks/wb.py:155)
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({"knn/top1": top1}, step=epoch)
        except ImportError:
            pass

    return _callback
