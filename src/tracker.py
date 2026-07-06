"""
tracker.py — Lightweight IoU-based multi-object tracker.

Assigns a persistent integer track ID to each detected person across
frames, so downstream logic (crowd clustering, snapshot events) can
reason about WHO is in a group instead of just WHERE a group is.

This is a SORT-style tracker without a re-identification / appearance
model: matching is done purely on bounding-box IoU (Intersection over
Union) between frames via the Hungarian algorithm. It's cheap, has no
extra ML model to load, and is a solid fit for a single fixed camera
where people don't jump across the frame between frames.

Trade-offs to be aware of:
  - IDs are NOT guaranteed stable across a long occlusion (someone
    walking behind another person / out of frame and back). If a track
    isn't matched for `max_age` frames it is deleted; the person gets a
    new ID when re-detected.
  - No appearance re-identification: if two people cross paths and
    swap positions in the same frame, ID swaps are possible in rare
    cases. For a "who is in this crowd" signal (not forensic identity),
    this is an acceptable, standard trade-off.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU matrix between two sets of [x1,y1,x2,y2] boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = boxes_a[:, None, :]   # (A,1,4)
    b = boxes_b[None, :, :]   # (1,B,4)

    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.clip(x2 - x1, 0, None)
    inter_h = np.clip(y2 - y1, 0, None)
    inter   = inter_w * inter_h

    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    union  = area_a + area_b - inter

    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


class Track:
    __slots__ = ("id", "box", "score", "hits", "time_since_update")

    def __init__(self, track_id, box, score):
        self.id                = track_id
        self.box                = box
        self.score              = score
        self.hits               = 1
        self.time_since_update  = 0


class IOUTracker:
    def __init__(self, cfg: dict):
        t_cfg = cfg.get("tracker", {})
        self.iou_threshold = t_cfg.get("iou_threshold", 0.3)
        self.max_age       = t_cfg.get("max_age", 15)   # frames a track survives unmatched
        self.min_hits      = t_cfg.get("min_hits", 1)   # frames before a track is "confirmed"

        self._tracks  = []   # list[Track]
        self._next_id = 1

    def update(self, detections):
        """
        detections: list of (box[x1,y1,x2,y2] as np.ndarray, score)
        Returns: list of (track_id, box, score) for confirmed, currently-matched tracks.
        """
        det_boxes  = np.array([d[0] for d in detections], dtype=np.float32) if detections else np.empty((0, 4), dtype=np.float32)
        det_scores = [d[1] for d in detections]

        track_boxes = np.array([t.box for t in self._tracks], dtype=np.float32) if self._tracks else np.empty((0, 4), dtype=np.float32)

        matched_track_idx = set()
        matched_det_idx   = set()

        if len(self._tracks) > 0 and len(detections) > 0:
            iou_matrix = iou_batch(track_boxes, det_boxes)
            row_idx, col_idx = linear_sum_assignment(-iou_matrix)  # maximize IoU

            for r, c in zip(row_idx, col_idx):
                if iou_matrix[r, c] >= self.iou_threshold:
                    track = self._tracks[r]
                    track.box               = det_boxes[c]
                    track.score              = det_scores[c]
                    track.hits              += 1
                    track.time_since_update  = 0
                    matched_track_idx.add(r)
                    matched_det_idx.add(c)

        # Age out unmatched tracks
        for i, track in enumerate(self._tracks):
            if i not in matched_track_idx:
                track.time_since_update += 1

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        # Start new tracks for unmatched detections
        for c in range(len(detections)):
            if c not in matched_det_idx:
                self._tracks.append(Track(self._next_id, det_boxes[c], det_scores[c]))
                self._next_id += 1

        # Return currently-matched, confirmed tracks
        results = []
        for t in self._tracks:
            if t.time_since_update == 0 and t.hits >= self.min_hits:
                results.append((t.id, t.box, t.score))
        return results
