# Crowd Detection + Heatmap

Real-time people detection with Gaussian heatmap overlay using Faster R-CNN ResNet50 and FFmpeg NVDEC GPU decode.

## Folder Structure

```
crowd_detection/
├── main.py           # Entry point — display loop
├── capture.py        # Thread 1: FFmpeg NVDEC RTSP capture
├── inference.py      # Thread 2: Faster R-CNN GPU inference
├── heatmap.py        # Gaussian heatmap helpers
├── config.yaml       # All settings (camera, model, heatmap, display)
├── requirements.txt
└── README.md
```

## Architecture

```
Thread 1  RTSPCapture   FFmpeg NVDEC → always holds latest frame
Thread 2  InferWorker   Faster R-CNN → publishes detections
Main      Display loop  draws boxes + heatmap, never blocks on inference
```

## Install

```bash
pip install -r requirements.txt
```

FFmpeg with NVDEC support is also required:
```bash
# Check
ffmpeg -hwaccels | grep cuda
ffmpeg -codecs | grep cuvid
```

## Run

```bash
python main.py
# or with a custom config:
python main.py --config config.yaml
```

Press `q` to quit.

## Config (config.yaml)

| Section | Key | Description |
|---|---|---|
| camera | rtsp_url | Full RTSP URL of your camera |
| camera | width / height | Stream resolution |
| ffmpeg | hwaccel | `cuda` for GPU decode, `none` for CPU |
| ffmpeg | codec | `hevc_cuvid` or `h264_cuvid` |
| model | confidence_threshold | Detection confidence (0.0–1.0) |
| heatmap | decay | Fade speed — 0.9 fast, 0.99 slow |
| heatmap | colormap | JET / HOT / INFERNO / TURBO |
| display | show_boxes | Toggle bounding boxes |
| display | show_hud | Toggle FPS/people counter |
