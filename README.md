# Crowd Detection System

GPU-accelerated RTSP crowd detection: NVDEC capture → Faster R-CNN person
detection → IoU multi-object tracker (persistent IDs) → DBSCAN spatial
clustering → crowd alert, ID-aware snapshot events, optional heatmap
overlay.

## Folder structure

```
crowd_detection/
├── main.py               # entry point — run this
├── config.yaml            # all settings live here
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── capture.py         # RTSP + NVDEC frame capture thread
│   ├── inference.py       # Faster R-CNN worker thread
│   ├── tracker.py         # IoU-based tracker — gives each person a persistent ID
│   ├── clustering.py      # DBSCAN crowd clustering (ID-aware)
│   ├── heatmap.py         # Gaussian heatmap accumulate/render
│   └── snapshot.py        # saves a photo when a NEW crowd event appears
├── snapshots/              # crowd snapshots get written here (auto-created)
└── logs/                   # reserved for future logging
```

## Run

```bash
pip install -r requirements.txt
python main.py --config config.yaml
```

## Pipeline

```
frame -> Faster R-CNN detections -> IOUTracker (assigns track_id) ->
DBSCAN clustering (clusters carry track_ids) -> crowd alert / snapshot / heatmap
```

## Person tracking (src/tracker.py)

A lightweight SORT-style tracker (IoU + Hungarian matching, no
appearance/re-id model) assigns a persistent integer ID to each detected
person. This is what lets everything downstream reason about **who** is
in a group, not just where boxes happen to be.

```yaml
tracker:
  iou_threshold: 0.3   # min overlap to match a detection to an existing track
  max_age: 15           # frames a track survives with no match (~1s @ 15fps)
  min_hits: 1           # frames before a track is reported (1 = immediate)
```

Limitations to know about: IDs are not guaranteed stable across a long
occlusion (person leaves frame / is fully blocked for longer than
`max_age` frames gets a new ID on return), and there's no appearance
re-identification, so in rare cases of people crossing paths exactly at
the same spot an ID swap is possible. For "who is in this crowd" this is
a standard, acceptable trade-off — for forensic-grade identity you'd
need a full re-id model, which is a much heavier addition.

## Crowd snapshots (src/snapshot.py) — ID-based event tracking

Snapshots are saved per **crowd event**, and events are identified by
**which track IDs are in the group** (via Jaccard similarity of the
ID sets), not by where the group is standing. This fixes the exact
failure mode where a naive "is a crowd currently present" flag or a
distance-based check can't tell two different groups apart:

- group `{1,2}` forms → snapshot A
- `{1,2}` drifts apart; `{1,3}` forms nearby → correctly a NEW event → snapshot B
- `{2,3}` forms elsewhere → correctly a NEW event → snapshot C
- `{1,2}` re-forms and just stands there → no repeat snapshots (same event)

```yaml
snapshot:
  enabled: true
  mode: "event"                # "event" = capped shots per event | "interval" = periodic while active
  save_dir: "snapshots"
  cooldown_sec: 5               # min seconds between shots of the SAME event
  event_gap_sec: 10             # event expires if not seen again within this long
  max_snapshots_per_event: 1    # cap follow-up shots per event (0 = unlimited)
  min_overlap_ratio: 0.5        # Jaccard threshold: >=0.5 shared members = "same" event
  filename_format: "crowd_%Y%m%d_%H%M%S_{count}p.jpg"
  jpeg_quality: 90
```

`min_overlap_ratio` example: group `{1,2}` vs `{1,3}` → intersection={1},
union={1,2,3} → overlap = 1/3 ≈ 0.33 → below the 0.5 threshold → correctly
treated as a different event.

## Heatmap on/off toggle

```yaml
heatmap:
  enabled: true      # set to false to disable heatmap entirely (zero extra cost)
  decay: 0.94
  alpha: 0.5
  sigma: 40
  colormap: "JET"
```

## Display

```yaml
display:
  show_boxes: true
  show_scores: true
  show_ids: true      # shows each person's persistent tracker ID on their box
  show_hud: true
```

## Notes
- `snapshots/` and `logs/` are created automatically if missing.
- All thresholds (confidence, crowd size, distance, tracker, snapshot)
  remain in `config.yaml` — no code changes needed for tuning.
