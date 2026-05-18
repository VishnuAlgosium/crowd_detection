"""
main.py — Crowd Detection + Heatmap entry point

Crowd detection uses DBSCAN spatial clustering —
only triggers when people are physically close together in the frame.

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
from sklearn.cluster import DBSCAN

from capture   import RTSPCapture
from inference import InferWorker, load_model
from heatmap   import add_heat, render_heatmap


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

quit_key      = ord(disp_cfg.get("quit_key", "q"))
crowd_trigger = crowd_cfg["min_people"]
max_distance  = crowd_cfg["max_distance"]   # pixels — how close = "together"
box_color     = tuple(crowd_cfg["box_color"])
label_color   = tuple(crowd_cfg["label_color"])
crowd_label   = crowd_cfg["label"]


# ─────────────────────────────────────────
# CLUSTERING HELPER
# ─────────────────────────────────────────

def find_crowd_clusters(detections, min_people, max_dist):
    """
    Returns list of clusters, each cluster is a list of (box, score).
    Only clusters with >= min_people are returned.

    Uses foot-centre (bottom-centre of box) as the person's position.
    """
    if len(detections) < min_people:
        return []

    # Foot-centre points for each detection
    points = np.array([
        [(b[0] + b[2]) / 2, b[3]]   # cx, foot_y
        for b, _ in detections
    ], dtype=np.float32)

    # DBSCAN — groups points within max_dist pixels of each other
    db     = DBSCAN(eps=max_dist, min_samples=min_people).fit(points)
    labels = db.labels_   # -1 = noise (isolated person)

    clusters = []
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue   # skip isolated people
        members = [detections[i] for i, l in enumerate(labels) if l == cluster_id]
        if len(members) >= min_people:
            clusters.append(members)

    return clusters


# ─────────────────────────────────────────
# START CAPTURE + MODEL + INFERENCE
# ─────────────────────────────────────────

print("[Main] Starting capture...")
cap = RTSPCapture(cfg)
time.sleep(2.0)

cam_w, cam_h, cam_fps = cap.get_props()
print(f"[Main] Camera: {cam_w}x{cam_h} @ {cam_fps} FPS")

model, transform, device = load_model(cfg)
worker = InferWorker(model, transform, device, cap, cfg)

heatmap   = np.zeros((cam_h, cam_w), dtype=np.float32)
fps_times = deque(maxlen=30)

print(f"[Main] Crowd alert: >= {crowd_trigger} people within {max_distance}px of each other")
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

    detections   = worker.get_detections()
    person_count = len(detections)

    # ── Spatial crowd clustering ─────────────────────────────────
    crowd_clusters = find_crowd_clusters(detections, crowd_trigger, max_distance)
    is_crowd       = len(crowd_clusters) > 0

    # ── Heatmap decay ───────────────────────────────────────────
    heatmap *= heat_cfg["decay"]

    # ── Draw individual person boxes ────────────────────────────
    for (x1, y1, x2, y2), score in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        add_heat(heatmap, (x1 + x2) // 2, y2, sigma=heat_cfg["sigma"])

        if disp_cfg["show_boxes"]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if disp_cfg["show_scores"]:
            cv2.putText(frame, f"{score:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2)

    # ── Draw crowd cluster boxes ─────────────────────────────────
    for cluster in crowd_clusters:
        boxes = [b for b, _ in cluster]

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

    # ── Red banner if any crowd cluster found ────────────────────
    if is_crowd:
        total_in_crowds = sum(len(c) for c in crowd_clusters)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (cam_w, 50), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame,
                    f"CROWD ALERT  —  {total_in_crowds} people in {len(crowd_clusters)} group(s)",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    # ── Blend heatmap ───────────────────────────────────────────
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