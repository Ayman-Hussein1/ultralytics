"""Cosine schedule for AdamW weight_decay across training epochs.

In DINOv3 / EUPE / DINOv2 student-distillation recipes the weight_decay coefficient is *not*
fixed: it ramps from a small `start` value (encouraging the network to fit the teacher early)
toward a larger `peak/end` value (regularising late-stage training to avoid overfit). Our
fastvit-s × 7-source distill collapses to chance-level kNN by ep17 with fixed wd=0.02,
matching the failure mode the schedules below were designed to prevent.

Reference recipes that motivated this callback:
  - DINOv3 ConvNeXt-Tiny distill: ``dinov3/configs/train/distillation_convnext/convnext_tiny_p16.yaml``
    schedules.weight_decay: start=0.04, peak=0.2, end=0.2, warmup_epochs=500 (i.e. linearly
    increases over the entire 500-ep run; effectively cosine-equivalent given wd_end=peak).
  - EUPE SSL ``eupe/configs/ssl_default_config.yaml`` optim block:
    weight_decay=0.04, weight_decay_end=0.4 (10× ramp, monotonic).
  - DINOv2 paper §A.3 "Hyper-parameters" (Oquab et al., 2023): "weight decay follows a cosine
    schedule from 0.04 to 0.4."

We use a half-cosine (raised cosine) so wd interpolates smoothly between start and end,
matching DINOv2's published shape and DINOv3's effective shape for long warmup_epochs. This
generalises the linear schedule used by the published configs without changing the endpoints.

Usage:
    from callbacks import wd_schedule
    model.add_callback("on_train_epoch_start", wd_schedule.override(start=0.04, end=0.2))

Notes:
  - Updates ``optimizer.param_groups[i]["weight_decay"]`` directly each epoch start. AdamW
    re-reads pg["weight_decay"] every step (PyTorch source: torch.optim.adamw._single_tensor_adamw),
    so per-epoch updates suffice — no need to hook every step.
  - Skips param groups whose existing ``weight_decay == 0`` (typical for biases/norm params under
    ``optimizer.add_param_group(... wd=0)`` convention; those should stay unregularised).
"""

from ultralytics.utils.torch_utils import one_cycle


def override(start=0.04, end=0.2):
    """Return on_train_epoch_start callback that scales AdamW weight_decay via half-cosine.

    Args:
        start (float): Initial weight_decay at epoch 0. DINOv3/EUPE/DINOv2 use 0.04.
        end (float): Final weight_decay at the last epoch. DINOv3 uses 0.2; EUPE uses 0.4.

    Notes:
        Half-cosine interpolation between start and end across ``trainer.epochs``. At epoch 0
        wd = start; at epoch ``trainer.epochs - 1`` wd ≈ end. Param groups initialised with
        weight_decay=0 (biases/norms by Ultralytics convention) are left untouched so the
        schedule only affects regularised parameter groups.
    """

    def callback(trainer):
        # Half-cosine via ``ultralytics.utils.torch_utils.one_cycle`` (the same primitive used
        # for LR scheduling): at epoch=0 returns ``start``, at epoch=epochs-1 returns ``end``.
        # Matches DINOv2 §A.3 schedule shape (cosine from 0.04 to 0.4).
        wd = one_cycle(start, end, max(trainer.epochs - 1, 1))(trainer.epoch)
        for pg in trainer.optimizer.param_groups:
            # Convention from Ultralytics's optimizer build: bias/norm groups get wd=0 and
            # must stay unregularised (matches DINOv3 ``layerwise_decay=1.0`` + WD-only-on-weights
            # convention).
            if pg["weight_decay"] > 0.0:
                pg["weight_decay"] = wd

    return callback
