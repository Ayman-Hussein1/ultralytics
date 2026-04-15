# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""FastTracker: Occlusion-aware multi-object tracker with mean history recovery and IoU-based init suppression."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils.ops import xywh2ltwh
from .basetrack import BaseTrack, TrackState
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYAH


class FastSTrack(BaseTrack):
    """Single object track with occlusion-aware state management and mean history for recovery.

    Extends the base tracking with occlusion detection flags and a rolling history of Kalman filter
    states that enables velocity/position reset during occlusion events.

    Attributes:
        shared_kalman (KalmanFilterXYAH): Shared Kalman filter used across all FastSTrack instances.
        mean_history (list[np.ndarray]): Rolling history of Kalman mean states (max 100 entries).
        not_matched (int): Consecutive frames where this track was not matched to a detection.
        is_occluded (bool): Whether the track is currently detected as occluded.
        occluded_len (int): Number of consecutive frames the track has been occluded.
        last_occluded_frame (int): Frame ID when the track was last detected as occluded.
        was_recently_occluded (bool): Flag indicating recent occlusion history.
    """

    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh: list[float], score: float, cls: Any):
        """Initialize a new FastSTrack instance.

        Args:
            xywh (list[float]): Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format.
            score (float): Confidence score of the detection.
            cls (Any): Class label for the detected object.
        """
        super().__init__()
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None

        # Occlusion-aware attributes
        self.mean_history: list[np.ndarray] = []
        self.not_matched = 0
        self.is_occluded = False
        self.occluded_len = 0
        self.last_occluded_frame = -1
        self.was_recently_occluded = False

    def _append_history(self):
        """Append current Kalman mean to rolling history (max 100 entries)."""
        self.mean_history.append(self.mean.copy())
        if len(self.mean_history) > 100:
            self.mean_history.pop(0)

    def predict(self):
        """Predict the next state (mean and covariance) of the object using the Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: list[FastSTrack]):
        """Perform multi-object predictive tracking using Kalman filter for the provided list of tracks."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = FastSTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilterXYAH, frame_id: int):
        """Activate a new tracklet using the provided Kalman filter and initialize its state."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))
        self._append_history()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: FastSTrack, frame_id: int, new_id: bool = False):
        """Reactivate a previously lost track using new detection data."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self._append_history()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def update(self, new_track: FastSTrack, frame_id: int):
        """Update the state of a matched track with new detection data."""
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self._append_history()
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self) -> np.ndarray:
        """Get the bounding box in top-left-width-height format from the current state estimate."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        """Convert bounding box from tlwh to (min x, min y, max x, max y) format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding box from tlwh format to center-x-center-y-aspect-height (xyah) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def xywh(self) -> np.ndarray:
        """Get the current position of the bounding box in (center x, center y, width, height) format."""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self) -> np.ndarray:
        """Get position in (center x, center y, width, height, angle) format."""
        if self.angle is None:
            from ..utils import LOGGER

            LOGGER.warning("`angle` attr not found, returning `xywh` instead.")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self) -> list[float]:
        """Get the current tracking results in the appropriate bounding box format."""
        coords = self.xyxy if self.angle is None else self.xywha
        return [*coords.tolist(), self.track_id, self.score, self.cls, self.idx]

    def __repr__(self) -> str:
        """Return a string representation of the FastSTrack object."""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
    inter_x1, inter_y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    inter_x2, inter_y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _is_occluded_by(box_a: np.ndarray, box_b: np.ndarray, iou_thresh: float = 0.7) -> bool:
    """Check if box_a is significantly overlapped by box_b using intersection-over-area-of-a."""
    inter_x1, inter_y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    inter_x2, inter_y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    if area_a == 0:
        return False
    return (inter / area_a) > iou_thresh


class FASTTRACK:
    """FastTracker: Occlusion-aware multi-object tracker built on the ByteTrack association strategy.

    Extends the two-stage detection association from ByteTrack with occlusion detection and handling,
    mean history-based state recovery during occlusions, and IoU-based initialization suppression
    to prevent duplicate tracks from overlapping detections.

    Attributes:
        tracked_stracks (list[FastSTrack]): List of successfully activated tracks.
        lost_stracks (list[FastSTrack]): List of lost tracks.
        removed_stracks (list[FastSTrack]): List of removed tracks.
        frame_id (int): The current frame ID.
        args (Namespace): Tracker configuration arguments.
        max_time_lost (int): Maximum frames for a track to be considered as 'lost'.
        kalman_filter (KalmanFilterXYAH): Kalman Filter object.
        occ_iou_thresh (float): IoU threshold to detect occlusion between tracks.
        reset_velocity_offset (int): Frames back to reset velocity from mean history.
        reset_position_offset (int): Frames back to reset position from mean history.
        enlarge_bbox_factor (float): Factor to enlarge bbox height on first occlusion frame.
        dampen_motion_factor (float): Velocity dampening factor during occlusion.
        occ_to_lost_thresh (int): Frames occluded before marking as lost.
        recent_occ_window (int): Frames after occlusion to still consider as recently occluded.
        init_iou_suppress (float): IoU threshold to suppress duplicate track initialization.
    """

    def __init__(self, args, frame_rate: int = 30):
        """Initialize a FASTTRACK instance for occlusion-aware object tracking.

        Args:
            args (Namespace): Command-line arguments containing tracking parameters.
            frame_rate (int): Frame rate of the video sequence.
        """
        self.tracked_stracks: list[FastSTrack] = []
        self.lost_stracks: list[FastSTrack] = []
        self.removed_stracks: list[FastSTrack] = []

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

        # Occlusion handling parameters
        self.occ_iou_thresh = getattr(args, "occ_iou_thresh", 0.7)
        self.reset_velocity_offset = getattr(args, "reset_velocity_offset", 5)
        self.reset_position_offset = getattr(args, "reset_position_offset", 3)
        self.enlarge_bbox_factor = getattr(args, "enlarge_bbox_factor", 1.1)
        self.dampen_motion_factor = getattr(args, "dampen_motion_factor", 0.89)
        self.occ_to_lost_thresh = getattr(args, "occ_to_lost_thresh", 10)
        self.recent_occ_window = getattr(args, "recent_occ_window", 40)
        self.init_iou_suppress = getattr(args, "init_iou_suppress", 0.7)

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
        """Update the tracker with new detections and return tracked objects.

        Implements a two-stage association (high/low confidence) like ByteTrack, plus occlusion-aware
        handling for unmatched tracks and IoU-based suppression for new track initialization.

        Args:
            results: Detection results with conf, xywh/xywhr, and cls attributes.
            img (np.ndarray | None): Current frame image (unused, kept for interface compatibility).
            feats (np.ndarray | None): Feature embeddings (unused, kept for interface compatibility).

        Returns:
            (np.ndarray): Array of tracked object results with format [x1, y1, x2, y2, id, score, cls, idx].
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_low & inds_high
        results_second = results[inds_second]
        results_keep = results[remain_inds]

        detections = self.init_track(results_keep)

        # Separate unconfirmed and confirmed tracks
        unconfirmed = []
        tracked_stracks: list[FastSTrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Step 2: First association with high score detection boxes
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)

        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            # Clear occlusion state on successful match
            track.is_occluded = False
            track.not_matched = 0
            track.occluded_len = 0

        # Step 3: Second association with low score detection boxes
        detections_second = self.init_track(results_second)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections_second)
        matches, u_track_second, _u_detection_second = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            # Clear occlusion state on successful match
            track.is_occluded = False
            track.not_matched = 0
            track.occluded_len = 0

        # Occlusion handling for unmatched tracks after second association
        for it in u_track_second:
            track = r_tracked_stracks[it]
            track.not_matched += 1

            # Try detecting occlusion against successfully matched tracks
            if not track.is_occluded and track.state == TrackState.Tracked:
                for other in activated_stracks:
                    if track.track_id == other.track_id:
                        continue
                    if not other.is_activated or other.is_occluded:
                        continue
                    if _is_occluded_by(track.xyxy, other.xyxy, self.occ_iou_thresh):
                        track.is_occluded = True
                        track.occluded_len += 1
                        track.last_occluded_frame = self.frame_id
                        track.was_recently_occluded = True

                        # Reset velocity from mean history
                        if len(track.mean_history) >= self.reset_velocity_offset:
                            old_mean = track.mean_history[-self.reset_velocity_offset]
                            track.mean[4:8] = old_mean[4:8]

                        # Reset position from mean history
                        if len(track.mean_history) >= self.reset_position_offset:
                            old_mean = track.mean_history[-self.reset_position_offset]
                            track.mean[0:4] = old_mean[0:4]

                        # Enlarge bbox on first occlusion frame
                        if track.occluded_len == 1:
                            track.mean[3] *= self.enlarge_bbox_factor

                        # Dampen motion
                        track.mean[4:8] *= self.dampen_motion_factor
                        break

            # Update occlusion counters
            if not track.is_occluded:
                track.occluded_len = 0
            else:
                track.occluded_len += 1

            if track.was_recently_occluded and (self.frame_id - track.last_occluded_frame > self.recent_occ_window):
                track.was_recently_occluded = False

            # Decide whether to mark as lost
            if track.state != TrackState.Lost:
                if track.not_matched > 2 and (
                    not track.is_occluded or track.occluded_len > self.occ_to_lost_thresh
                ):
                    track.mark_lost()
                    lost_stracks.append(track)

        # Deal with unconfirmed tracks
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Step 4: Init new stracks with IoU suppression
        active_now = {t.track_id: t for t in self.tracked_stracks if t.state == TrackState.Tracked}
        for t in activated_stracks:
            active_now[t.track_id] = t
        active_tracks = list(active_now.values())

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue

            # Check max IoU with any active track - suppress if too high
            det_box = track.xyxy
            suppress = False
            for at in active_tracks:
                if _box_iou(det_box, at.xyxy) >= self.init_iou_suppress:
                    suppress = True
                    break

            if not suppress:
                track.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(track)

        # Step 5: Update state - handle lost tracks with occlusion awareness
        for track in self.lost_stracks:
            recently_occluded = track.was_recently_occluded and (
                self.frame_id - track.last_occluded_frame <= self.recent_occ_window
            )
            if not recently_occluded and (self.frame_id - track.end_frame > self.max_time_lost):
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]

        # Only return tracks updated this frame (have valid idx for current detections).
        # Occluded tracks kept alive internally but without a matching detection are excluded
        # from the output since there is no corresponding result to map back to.
        return np.asarray(
            [x.result for x in self.tracked_stracks if x.is_activated and x.frame_id == self.frame_id],
            dtype=np.float32,
        )

    def get_kalmanfilter(self) -> KalmanFilterXYAH:
        """Return a Kalman filter object for tracking bounding boxes using KalmanFilterXYAH."""
        return KalmanFilterXYAH()

    def init_track(self, results, img: np.ndarray | None = None) -> list[FastSTrack]:
        """Initialize object tracking with given detections as FastSTrack instances."""
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        return [FastSTrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def get_dists(self, tracks: list[FastSTrack], detections: list[FastSTrack]) -> np.ndarray:
        """Calculate the distance between tracks and detections using IoU and optionally fuse scores."""
        dists = matching.iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks: list[FastSTrack]):
        """Predict the next states for multiple tracks using Kalman filter."""
        FastSTrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """Reset the ID counter for FastSTrack instances."""
        FastSTrack.reset_id()

    def reset(self):
        """Reset the tracker by clearing all tracked, lost, and removed tracks."""
        self.tracked_stracks: list[FastSTrack] = []
        self.lost_stracks: list[FastSTrack] = []
        self.removed_stracks: list[FastSTrack] = []
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    @staticmethod
    def joint_stracks(tlista: list[FastSTrack], tlistb: list[FastSTrack]) -> list[FastSTrack]:
        """Combine two lists of FastSTrack objects into a single list, ensuring no duplicates."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista: list[FastSTrack], tlistb: list[FastSTrack]) -> list[FastSTrack]:
        """Filter out the stracks present in the second list from the first list."""
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(
        stracksa: list[FastSTrack], stracksb: list[FastSTrack]
    ) -> tuple[list[FastSTrack], list[FastSTrack]]:
        """Remove duplicate stracks from two lists based on IoU distance."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
