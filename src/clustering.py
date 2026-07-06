"""
clustering.py — DBSCAN spatial clustering for crowd detection
Only triggers when people are physically close together in the frame.

Works on TRACKED detections (track_id, box, score) so each resulting
cluster carries the set of persistent track IDs in it — this is what
lets snapshot.py tell "group {1,2}" apart from "group {1,3}" reliably,
instead of guessing from position alone.
"""

import numpy as np
from sklearn.cluster import DBSCAN


def find_crowd_clusters(tracked_detections, min_people, max_dist):
    """
    tracked_detections: list of (track_id, box[x1,y1,x2,y2], score)

    Returns list of clusters; each cluster is a list of
    (track_id, box, score) — same tuple shape as the input — for every
    member of that cluster. Only clusters with >= min_people are kept.

    Uses foot-centre (bottom-centre of box) as the person's position.
    """
    if len(tracked_detections) < min_people:
        return []

    points = np.array([
        [(b[0] + b[2]) / 2, b[3]]   # cx, foot_y
        for _, b, _ in tracked_detections
    ], dtype=np.float32)

    db     = DBSCAN(eps=max_dist, min_samples=min_people).fit(points)
    labels = db.labels_   # -1 = noise (isolated person)

    clusters = []
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        members = [tracked_detections[i] for i, l in enumerate(labels) if l == cluster_id]
        if len(members) >= min_people:
            clusters.append(members)

    return clusters
