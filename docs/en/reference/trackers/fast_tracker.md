---
description: Explore the FastTracker module in Ultralytics — an occlusion-aware ByteTrack-style multi-object tracker with lightweight Kalman rollback, motion dampening during occlusion, and init-IoU suppression to reduce duplicate tracks.
keywords: Ultralytics, FastTracker, FASTTracker, FastSTrack, occlusion-aware tracking, ByteTrack, Kalman filter, multi-object tracking, MOT
---

# Reference for `ultralytics/trackers/fast_tracker.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/fast_tracker.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/fast_tracker.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

FastTracker ([arXiv:2508.14370](https://arxiv.org/abs/2508.14370)) is an occlusion-aware extension of ByteTrack designed for real-time use. When a track becomes occluded (measured by the `occ_cover_thresh` overlap test), FastTracker **rolls back the Kalman state** to a recent pre-occlusion frame (`reset_pos_offset_occ`, `reset_velocity_offset_occ`), **dampens motion** (`dampen_motion_occ`), and **enlarges the search bbox** (`enlarge_bbox_occ`) so the track can be re-acquired without drifting. A short re-appearance window (`occ_reappear_window`) keeps recently-occluded lost tracks re-findable, and **init-IoU suppression** (`init_iou_suppress`) blocks new-track initialization on top of an already-active track to reduce duplicates. Enable it with `tracker="fasttrack.yaml"`.

<br>

## ::: ultralytics.trackers.fast_tracker.FastSTrack

<br><br><hr><br>

## ::: ultralytics.trackers.fast_tracker.FASTTracker

<br><br>
