---
description: Explore the OC-SORT tracker in Ultralytics — observation-centric multi-object tracking with velocity direction consistency (OCM), observation-centric re-update (OCR), and an optional ByteTrack-style low-confidence association pass.
keywords: Ultralytics, OC-SORT, OCSORT, OCSortTrack, observation-centric, object tracking, Kalman filter, OCM, OCR, MOT
---

# Reference for `ultralytics/trackers/oc_sort.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/oc_sort.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

OC-SORT (Cao et al., [arXiv:2203.14360](https://arxiv.org/abs/2203.14360)) extends the SORT family with three observation-centric ideas: **Observation-Centric Re-update (ORU)** to repair Kalman state after long occlusions using the last reliable observation, **Observation-Centric Momentum (OCM)** that adds a velocity-direction consistency term to the association cost, and **Observation-Centric Recovery (OCR)** for re-associating previously lost tracks against unmatched detections. Enable it with `tracker="ocsort.yaml"`.

<br>

## ::: ultralytics.trackers.oc_sort.OCSortTrack

<br><br><hr><br>

## ::: ultralytics.trackers.oc_sort.OCSORT

<br><br>
