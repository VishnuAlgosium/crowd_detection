"""
main.py — Crowd Detection + Heatmap entry point

Pipeline: RTSP capture -> Faster R-CNN person detection -> IoU tracker
(persistent track IDs) -> DBSCAN spatial clustering -> crowd alert,
ID-aware snapshot events, optional heatmap overlay.

Run:
    python main.py
    python main.py --config config.yaml
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
import yaml

from src.capture    import RTSPCapture
from src.inference  import InferWorker, load_model
from src.tracker    import IOUTracker
from src.clustering import find_crowd_clusters
from src.snapshot   import SnapshotSaver
from src.heatmap    import add_heat, render_heatmap


# ─────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()


# ─────────────────────────────────────────
# LOAD CONFIG
# ─────────────────────────────────────────

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

disp_cfg  = cfg["display"]
heat_cfg  = cfg["heatmap"]
crowd_cfg = cfg["crowd"]

quit_key       = ord(disp_cfg.get("quit_key", "q"))
crowd_trigger  = crowd_cfg["min_people"]
max_distance   = crowd_cfg["max_distance"]   # pixels — how close = "together"
box_color      = tuple(crowd_cfg["box_color"])
label_color    = tuple(crowd_cfg["label_color"])
crowd_label    = crowd_cfg["label"]
heatmap_on     = heat_cfg.get("enabled", True)
show_ids       = disp_cfg.get("show_ids", True)


# ─────────────────────────────────────────
# START CAPTURE + MODEL + INFERENCE + TRACKER + SNAPSHOTS
# ─────────────────────────────────────────

print("[Main] Starting capture...")
cap = RTSPCapture(cfg)
time.sleep(2.0)

cam_w, cam_h, cam_fps = cap.get_props()
print(f"[Main] Camera: {cam_w}x{cam_h} @ {cam_fps} FPS")

model, transform, device = load_model(cfg)
worker   = InferWorker(model, transform, device, cap, cfg)
tracker  = IOUTracker(cfg)
snapshot = SnapshotSaver(cfg)

heatmap   = np.zeros((cam_h, cam_w), dtype=np.float32)
fps_times = deque(maxlen=30)

print(f"[Main] Crowd alert: >= {crowd_trigger} people within {max_distance}px of each other")
print(f"[Main] Heatmap: {'ON' if heatmap_on else 'OFF'}  |  Snapshots: {'ON' if snapshot.enabled else 'OFF'} "
      f"({snapshot.mode} mode, ID-based event tracking)")
print(f"[Main] Running — press '{disp_cfg.get('quit_key', 'q')}' to quit")


# ─────────────────────────────────────────
# MAIN DISPLAY LOOP
# ─────────────────────────────────────────

while True:
    t0 = time.perf_counter()

    ok, frame = cap.read()
    if not ok or frame is None:
        time.sleep(0.005)
        continue

    raw_detections = worker.get_detections()          # [(box, score), ...]
    tracked        = tracker.update(raw_detections)    # [(track_id, box, score), ...]
    person_count   = len(tracked)

    # ── Spatial crowd clustering (now ID-aware) ──────────────────
    crowd_clusters = find_crowd_clusters(tracked, crowd_trigger, max_distance)
    is_crowd       = len(crowd_clusters) > 0

    # ── Heatmap decay ───────────────────────────────────────────
    if heatmap_on:
        heatmap *= heat_cfg["decay"]

    # ── Draw individual person boxes ────────────────────────────
    for track_id, (x1, y1, x2, y2), score in tracked:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if heatmap_on:
            add_heat(heatmap, (x1 + x2) // 2, y2, sigma=heat_cfg["sigma"])

        if disp_cfg["show_boxes"]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label_bits = []
        if show_ids:
            label_bits.append(f"ID{track_id}")
        if disp_cfg["show_scores"]:
            label_bits.append(f"{score:.2f}")
        if label_bits:
            cv2.putText(frame, " ".join(label_bits),
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2)

    # ── Draw crowd cluster boxes ─────────────────────────────────
    cluster_id_sets = []  # set of track_ids per cluster — for ID-based snapshot events
    for cluster in crowd_clusters:
        boxes = [b for _, b, _ in cluster]

        # Tight box around this specific cluster
        gx1 = int(min(b[0] for b in boxes))
        gy1 = int(min(b[1] for b in boxes))
        gx2 = int(max(b[2] for b in boxes))
        gy2 = int(max(b[3] for b in boxes))

        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), box_color, 3)
        cv2.putText(frame,
                    f"{crowd_label} ({len(cluster)})",
                    (gx1, gy1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, label_color, 2)

        cluster_id_sets.append({tid for tid, _, _ in cluster})

    # ── Red banner if any crowd cluster found ────────────────────
    total_in_crowds = sum(len(c) for c in crowd_clusters)
    if is_crowd:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (cam_w, 50), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame,
                    f"CROWD ALERT  —  {total_in_crowds} people in {len(crowd_clusters)} group(s)",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    # ── Save snapshot: events are tracked by WHO is in the group
    #    (track-ID overlap), so {1,2} vs {1,3} vs {2,3} are each
    #    correctly treated as distinct crowd events ─────────────────
    snapshot.maybe_save(frame, cluster_id_sets)

    # ── Blend heatmap ───────────────────────────────────────────
    if heatmap_on:
        frame = render_heatmap(
            frame, heatmap,
            alpha=heat_cfg["alpha"],
            colormap=heat_cfg["colormap"],
        )

    # ── FPS + HUD ────────────────────────────────────────────────
    t1 = time.perf_counter()
    fps_times.append(t1 - t0)
    fps = 1.0 / (sum(fps_times) / len(fps_times))

    if disp_cfg["show_hud"]:
        hud_y = 70 if is_crowd else 40
        cv2.putText(frame, f"FPS:    {fps:.1f}",
                    (20, hud_y),       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"People: {person_count}",
                    (20, hud_y + 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),   2)
        cv2.putText(frame, f"Crowd:  {'YES (%d groups)' % len(crowd_clusters) if is_crowd else 'No'}",
                    (20, hud_y + 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if is_crowd else (200, 200, 200), 2)
        cv2.putText(frame, f"Res:    {cam_w}x{cam_h}",
                    (20, hud_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow(disp_cfg["window_title"], frame)

    if cv2.waitKey(1) & 0xFF == quit_key:
        break


# ─────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────

worker.stop()
cap.release()
cv2.destroyAllWindows()
print("[Main] Done")
