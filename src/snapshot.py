"""
snapshot.py — Saves a snapshot image when a NEW crowd event is detected.

Events are identified by WHO is in the crowd, using the persistent
track IDs from src/tracker.py — not by where the group is standing.
Each crowd cluster carries a set of track IDs (e.g. {1, 2}). A cluster
is considered a continuation of an already-active event if its track-ID
set overlaps an active event's set by at least `min_overlap_ratio`
(Jaccard similarity: |intersection| / |union|). Otherwise it's treated
as a brand-new event and gets its own snapshot immediately.

This correctly tells these apart, which pure distance/position matching
could not:
  - group {1,2} together, then {1,3} forms nearby        -> 2 events
  - group {1,2} together, then {2,3} forms nearby        -> 2 events
  - group {1,2} drifts across the whole frame, unchanged -> 1 event

Modes (config: snapshot.mode):
  "event"    (default) — up to `max_snapshots_per_event` shots per
             tracked event.
  "interval" — one shot every `cooldown_sec` per tracked event, no cap.
"""

import os
import time
from datetime import datetime

import cv2


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class SnapshotSaver:
    def __init__(self, cfg: dict):
        snap_cfg = cfg.get("snapshot", {})

        self.enabled          = snap_cfg.get("enabled", True)
        self.mode             = snap_cfg.get("mode", "event")     # "event" | "interval"
        self.save_dir         = snap_cfg.get("save_dir", "snapshots")
        self.cooldown_sec     = snap_cfg.get("cooldown_sec", 5)
        self.event_gap_sec    = snap_cfg.get("event_gap_sec", 10)    # event expires if unseen this long
        self.max_per_event    = snap_cfg.get("max_snapshots_per_event", 1)  # 0 = unlimited
        self.min_overlap_ratio = snap_cfg.get("min_overlap_ratio", 0.5)     # Jaccard threshold = "same" event
        self.filename_fmt     = snap_cfg.get(
            "filename_format", "crowd_%Y%m%d_%H%M%S_{count}p.jpg"
        )
        self.jpeg_quality     = snap_cfg.get("jpeg_quality", 90)

        self._active_events = []  # each: {ids:set, last_seen, last_saved, shots}

        if self.enabled:
            os.makedirs(self.save_dir, exist_ok=True)

    def maybe_save(self, frame, clusters):
        """
        Call every frame.
        `clusters`: list of frozenset/set of track_ids — one entry per
        crowd cluster detected THIS frame (empty list if no crowd).
        Returns list of file paths saved this call (may be empty).
        """
        if not self.enabled:
            return []

        now = time.time()

        # Expire events we haven't seen matched in a while
        self._active_events = [
            e for e in self._active_events
            if now - e["last_seen"] < self.event_gap_sec
        ]

        saved_paths = []

        for ids in clusters:
            ids = set(ids)
            match, best_overlap = None, 0.0
            for e in self._active_events:
                overlap = _jaccard(ids, e["ids"])
                if overlap >= self.min_overlap_ratio and overlap > best_overlap:
                    best_overlap, match = overlap, e

            if match is None:
                # New crowd event — no active event shares enough members
                match = {"ids": ids, "last_seen": now, "last_saved": 0.0, "shots": 0}
                self._active_events.append(match)
                should_save = True
            else:
                match["ids"]       = ids   # membership can drift a little frame to frame
                match["last_seen"] = now
                cooled_down = (now - match["last_saved"]) >= self.cooldown_sec
                if self.mode == "interval":
                    should_save = cooled_down
                else:  # "event" mode
                    under_cap = (self.max_per_event == 0 or
                                 match["shots"] < self.max_per_event)
                    should_save = under_cap and cooled_down

            if should_save:
                path = self._write(frame, len(ids), now)
                if path:
                    match["last_saved"] = now
                    match["shots"]    += 1
                    saved_paths.append(path)

        return saved_paths

    def _write(self, frame, person_count: int, now: float):
        timestamp = datetime.now()
        filename  = timestamp.strftime(self.filename_fmt).format(count=person_count)
        filepath  = os.path.join(self.save_dir, filename)

        ok = cv2.imwrite(
            filepath, frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )

        if ok:
            print(f"[Snapshot] Saved crowd snapshot -> {filepath}")
            return filepath

        print(f"[Snapshot] Failed to save snapshot to {filepath}")
        return None
