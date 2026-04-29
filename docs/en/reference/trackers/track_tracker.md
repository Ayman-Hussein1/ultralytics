---
description: Explore the TrackTrack module in Ultralytics — a multi-cue online multi-object tracker with HMIoU distance, iterative confidence-aware assignment, Track-Aware Initialization (TAI), and optional ReID and global motion compensation.
keywords: Ultralytics, TrackTrack, TTSTrack, HMIoU, iterative assignment, Track-Aware Initialization, TAI, ReID, GMC, multi-object tracking, MOT
---

# Reference for `ultralytics/trackers/track_tracker.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/track_tracker.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/track_tracker.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

TrackTrack ("Focusing on Tracks for Online Multi-Object Tracking", CVPR 2025) combines a multi-cue cost matrix — **HMIoU** distance, ReID cosine distance, confidence distance, and corner-angle distance, weighted by `iou_weight`, `reid_weight`, `conf_weight`, and `angle_weight` — with **iterative confidence-aware assignment** that progressively relaxes the matching threshold (`reduce_step`) while penalizing low-confidence and recovered detections (`penalty_p`, `penalty_q`). **Track-Aware Initialization (TAI)** uses NMS against existing tracks (`tai_thr`) plus a minimum history length (`min_track_len`) before confirming a new ID, reducing duplicate tracks. Optional ReID (`with_reid`) and sparse-optical-flow GMC are supported. Enable it with `tracker="tracktrack.yaml"`.

<br>

## ::: ultralytics.trackers.track_tracker.TTSTrack

<br><br><hr><br>

## ::: ultralytics.trackers.track_tracker.TRACKTRACK

<br><br>
